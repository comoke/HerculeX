#define _GNU_SOURCE
#include "cached_ct.h"

#include <fcntl.h>
#include <json-c/json.h>
#include <limits.h>
#include <stdio.h>

#include <dirent.h>
#include <glob.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>

#include "../../storage/data_roots.c"

#include "read_box.c"

static int cached_ct_vector(const char *text, int64_t vec[3]) {
    json_object *list = text ? json_tokener_parse(text) : NULL;
    int ok = json_object_is_type(list, json_type_array) && json_object_array_length(list) == 3;
    for (size_t i = 0; ok && i < 3; i++) {
        json_object *item = json_object_array_get_idx(list, i);
        vec[i] = json_object_get_int64(item);
        ok = json_object_is_type(item, json_type_int) && vec[i] > 0;
    }
    json_object_put(list);
    return ok;
}

static int cached_ct_safe(const char *name) {
    return name && *name && strcmp(name, ".") != 0 && strcmp(name, "..") != 0 && !strchr(name, '/');
}

static int cached_ct_no_wal(const char *path) {
    struct stat st;
    int status = lstat(path, &st);
    return status ? errno == ENOENT : S_ISREG(st.st_mode) && st.st_size == 0;
}

static int cached_ct_prepare(const char *path, sqlite3 **db, const char *sql, sqlite3_stmt **stmt) {
    int flags = SQLITE_OPEN_READONLY;
    int result = sqlite3_open_v2(path, db, flags, NULL);
    if (result == SQLITE_OK) {
        sqlite3_busy_timeout(*db, 60000);
        result = sqlite3_prepare_v2(*db, sql, -1, stmt, NULL);
    }
    if ((result & 255) != SQLITE_CANTOPEN && (result & 255) != SQLITE_READONLY) {
        return result;
    }
    char *wal = NULL;
    if (asprintf(&wal, "%s-wal", path) < 0) {
        return result;
    }
    if (cached_ct_no_wal(wal)) {
        char *uri = NULL;
        size_t len = strlen(path);
        char *encoded = len <= (SIZE_MAX - 1) / 3 ? malloc(len * 3 + 1) : NULL;
        if (encoded) {
            const char hex[] = "0123456789ABCDEF";
            char *out = encoded;
            for (const unsigned char *c = (const unsigned char *)path; *c; c++) {
                if (*c == '/') {
                    *out++ = '/';
                } else {
                    *out++ = '%';
                    *out++ = hex[*c >> 4];
                    *out++ = hex[*c & 15];
                }
            }
            *out = 0;
            uri = sqlite3_mprintf("file:%s?mode=ro&immutable=1", encoded);
            if (uri) {
                sqlite3_finalize(*stmt);
                *stmt = NULL;
                sqlite3_close(*db);
                *db = NULL;
                result = sqlite3_open_v2(uri, db, flags | SQLITE_OPEN_URI, NULL);
                if (result == SQLITE_OK) {
                    result = sqlite3_prepare_v2(*db, sql, -1, stmt, NULL);
                }
                if (result == SQLITE_OK && !cached_ct_no_wal(wal)) {
                    result = SQLITE_BUSY;
                }
            }
        }
        free(encoded);
        sqlite3_free(uri);
    }
    free(wal);
    return result;
}

void hx_cached_ct_close(HxCachedCt *ct) {
    if (ct->fd >= 0) {
        close(ct->fd);
    }
    sqlite3_close(ct->index);
    *ct = (HxCachedCt){.fd = -1};
}

int hx_cached_ct_open(
    HxCachedCt *ct, const char *root, const char *sample, const char *scan_or_url, int level) {
    *ct = (HxCachedCt){.fd = -1};
    char *path = NULL;
    if (asprintf(&path, "%s/arrays.sqlite", root) < 0) {
        return -1;
    }
    if (access(path, F_OK)) {
        int missing = errno == ENOENT || errno == ENOTDIR;
        free(path);
        return missing ? 1 : -1;
    }
    sqlite3 *registry = NULL;
    const char sql[] =
        "WITH candidate(src) AS (SELECT rtrim(url,'/') FROM resolved "
        "WHERE sample=?1 AND scan=?2 AND kind='ct') "
        "SELECT sample,name,shape,chunks,fill_value,dtype FROM arrays WHERE level=?3 AND "
        "(root=rtrim(?2,'/') OR url=rtrim(?2,'/') OR root=(SELECT src FROM candidate) "
        "OR url=(SELECT src FROM candidate) OR (sample=?1 AND (name=?2 OR instr(name,?2||'-')=1))) "
        "ORDER BY CASE WHEN root=rtrim(?2,'/') OR url=rtrim(?2,'/') OR root=(SELECT src FROM "
        "candidate) "
        "OR url=(SELECT src FROM candidate) THEN 0 WHEN name=?2 THEN 1 ELSE 2 END,name LIMIT 1";
    sqlite3_stmt *stmt = NULL;
    const char exact[] =
        "SELECT sample,name,shape,chunks,fill_value,dtype FROM arrays WHERE level=?3 "
        "AND root=rtrim(?2,'/')";
    int result = cached_ct_prepare(path, &registry, sample ? sql : exact, &stmt);
    free(path);
    path = NULL;
    if (result == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, sample, -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, scan_or_url, -1, SQLITE_STATIC);
        sqlite3_bind_int(stmt, 3, level);
        result = sqlite3_step(stmt);
    }
    int status = result == SQLITE_DONE ? 1 : -1;
    if (result == SQLITE_ROW) {
        const char *who = (const char *)sqlite3_column_text(stmt, 0);
        const char *name = (const char *)sqlite3_column_text(stmt, 1);
        const char *dtype = (const char *)sqlite3_column_text(stmt, 5);
        int64_t fill = sqlite3_column_int64(stmt, 4);
        int valid = dtype &&
            (!strcmp(dtype, "|u1") || !strcmp(dtype, "<u1") || !strcmp(dtype, "u1") ||
                !strcmp(dtype, "uint8")) &&
            fill >= 0 && fill <= UINT8_MAX && cached_ct_safe(who) && cached_ct_safe(name) &&
            cached_ct_vector((const char *)sqlite3_column_text(stmt, 2), ct->shape) &&
            cached_ct_vector((const char *)sqlite3_column_text(stmt, 3), ct->chunks);
        if (valid && asprintf(&path, "%s/%s/%s/%d/index.sqlite", root, who, name, level) >= 0) {
            ct->fill = (uint8_t)fill;
            sqlite3_stmt *probe = NULL;
            if (access(path, F_OK)) {
                status = errno == ENOENT || errno == ENOTDIR ? 1 : -1;
            } else if (cached_ct_prepare(path, &ct->index, "SELECT offset FROM chunks LIMIT 0",
                           &probe) == SQLITE_OK) {
                const char *leaf = strrchr(path, '/') + 1;
                char *pack = NULL;
                if (asprintf(&pack, "%.*schunks.pack", (int)(leaf - path), path) >= 0) {
                    ct->fd = open(pack, O_RDONLY | O_CLOEXEC);
                    status = ct->fd >= 0 || errno == ENOENT ? 0 : -1;
                    free(pack);
                }
            }
            sqlite3_finalize(probe);
        }
        free(path);
    }
    sqlite3_finalize(stmt);
    sqlite3_close(registry);
    if (status) {
        hx_cached_ct_close(ct);
    }
    return status;
}

int hx_cached_ct_read(HxCachedCt *ct, const int64_t box[6], uint8_t *pixels, int64_t *unknown) {
    int64_t first[3], count[3], low[3], high[3], size[3], valid[3];
    int64_t total = 1, bytes = 1, missing = 1, voxels = 1;
    for (int i = 0; i < 3; i++) {
        int64_t begin = box[2 * i], end = box[2 * i + 1], c = ct->chunks[i];
        if (c <= 0 || end <= begin || (begin < 0 && end > INT64_MAX + begin) ||
            begin < INT64_MIN + c || end > INT64_MAX - c || bytes > INT64_MAX / c) {
            return -1;
        }
        bytes *= c;
        first[i] = floor_div(begin, c);
        int64_t last = floor_div(end - 1, c) + 1;
        count[i] = last - first[i];
        size[i] = end - begin;
        if (total > INT64_MAX / count[i] || voxels > INT64_MAX / size[i]) {
            return -1;
        }
        total *= count[i];
        voxels *= size[i];
        low[i] = max64(0, first[i]);
        high[i] = min64(floor_div(ct->shape[i] - 1, c) + 1, last);
        missing *= max64(0, high[i] - low[i]);
        valid[i] = ct->shape[i] <= begin ? 0 : ct->shape[i] >= end ? size[i] : ct->shape[i] - begin;
    }
    int64_t *offset = (uint64_t)total <= SIZE_MAX / sizeof(*offset)
        ? malloc((size_t)total * sizeof(*offset))
        : NULL;
    if (!offset) {
        return -1;
    }
    for (int64_t i = 0; i < total; i++) {
        offset[i] = -1;
    }
    int64_t stored = 0;
    int ok = 1;
    if (missing) {
        const char sql[] = "SELECT z,y,x,offset FROM chunks "
                           "WHERE z>=?1 AND z<?2 AND y>=?3 AND y<?4 AND x>=?5 AND x<?6";
        sqlite3_stmt *stmt = NULL;
        ok = sqlite3_prepare_v2(ct->index, sql, -1, &stmt, NULL) == SQLITE_OK;
        for (int i = 0; ok && i < 3; i++) {
            sqlite3_bind_int64(stmt, 2 * i + 1, low[i]);
            sqlite3_bind_int64(stmt, 2 * i + 2, high[i]);
        }
        int step = SQLITE_ERROR;
        while (ok && (step = sqlite3_step(stmt)) == SQLITE_ROW) {
            int64_t z = sqlite3_column_int64(stmt, 0) - first[0];
            int64_t y = sqlite3_column_int64(stmt, 1) - first[1];
            int64_t x = sqlite3_column_int64(stmt, 2) - first[2];
            int64_t value = sqlite3_column_int64(stmt, 3);
            if (value >= -1) {
                offset[(z * count[1] + y) * count[2] + x] = value;
                missing--;
                stored += value >= 0;
            }
        }
        ok = ok && step == SQLITE_DONE;
        sqlite3_finalize(stmt);
    }
    int threads = stored < 4 ? 1 : (int)min64(6, stored);
    ok = ok && (!stored || ct->fd >= 0) &&
        !lg_read_box(ct->fd, ct->chunks, box, offset, ct->fill, pixels, threads);
    free(offset);
    if (!ok) {
        return -1;
    }
    if (valid[0] < size[0] || valid[1] < size[1] || valid[2] < size[2]) {
        for (int64_t z = 0; z < size[0]; z++) {
            for (int64_t y = 0; y < size[1]; y++) {
                int64_t x = z < valid[0] && y < valid[1] ? valid[2] : 0;
                memset(pixels + (z * size[1] + y) * size[2] + x, ct->fill, size[2] - x);
            }
        }
    }
    if (unknown) {
        *unknown = missing;
    }
    return 0;
}

static int cached_ct_count(const char *file, int64_t *stored, int64_t *absent) {
    sqlite3 *db = NULL;
    sqlite3_stmt *stmt = NULL;
    const char sql[] = "SELECT COALESCE(SUM(offset>=0),0),COALESCE(SUM(offset<0),0) FROM chunks";
    int ok =
        cached_ct_prepare(file, &db, sql, &stmt) == SQLITE_OK && sqlite3_step(stmt) == SQLITE_ROW;
    if (ok) {
        *stored = sqlite3_column_int64(stmt, 0);
        *absent = sqlite3_column_int64(stmt, 1);
    }
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return ok;
}

static char *cached_ct_root(void) {
    const char *value = getenv("HERCULEX_CHUNKS");
    if (value && *value) {
        return resolved(value);
    }
    char *outputs = local_root(outputs_root);
    char *path = outputs ? path_join(outputs, "cache/chunks") : NULL;
    char *root = path ? resolved(path) : NULL;
    free(outputs);
    free(path);
    return root;
}

int hx_chunk_usage_main(int argc, char **argv) {
    if (argc != 1 && (argc != 2 || strcmp(argv[1], "usage") != 0)) {
        fprintf(stderr, "Usage: ./run.sh scans.chunks.fetch usage\n");
        return 2;
    }
    char *root = cached_ct_root();
    char *pattern = root ? path_join(root, "*/*/*/index.sqlite") : NULL;
    char *registry_path = root ? path_join(root, "arrays.sqlite") : NULL;
    glob_t levels = {0};
    int listing = pattern ? glob(pattern, 0, NULL, &levels) : GLOB_ABORTED;
    int ok = registry_path && (listing == 0 || listing == GLOB_NOMATCH);
    sqlite3 *db = NULL;
    sqlite3_stmt *registered = NULL;
    if (ok && levels.gl_pathc && access(registry_path, F_OK) == 0) {
        const char sql[] = "SELECT 1 FROM arrays WHERE sample||'/'||name||'/'||level=?1";
        ok = cached_ct_prepare(registry_path, &db, sql, &registered) == SQLITE_OK;
    }
    printf("%-80s %8s %8s %8s\n", "array", "stored", "absent", "GB");
    int64_t total = 0;
    for (size_t i = 0; ok && i < levels.gl_pathc; i++) {
        const char *file = levels.gl_pathv[i], *leaf = strrchr(file, '/');
        char *folder = strndup(file, (size_t)(leaf - file));
        if (!folder) {
            ok = 0;
            break;
        }
        const char *key = folder + strlen(root) + 1;
        if (registered) {
            sqlite3_reset(registered);
            sqlite3_bind_text(registered, 1, key, -1, SQLITE_TRANSIENT);
        }
        int found = registered ? sqlite3_step(registered) : SQLITE_DONE;
        if (found != SQLITE_ROW) {
            ok = found == SQLITE_DONE;
            if (ok) {
                printf("%-80s (not in the registry)\n", key);
            }
            free(folder);
            continue;
        }
        int64_t stored = 0, absent = 0, bytes = 0;
        DIR *dir = opendir(folder);
        ok = dir && cached_ct_count(file, &stored, &absent);
        if (dir) {
            struct dirent *entry;
            while ((entry = readdir(dir))) {
                struct stat st;
                if (!fstatat(dirfd(dir), entry->d_name, &st, 0) && S_ISREG(st.st_mode)) {
                    bytes += st.st_size;
                }
            }
            closedir(dir);
        }
        if (ok) {
            total += bytes;
            printf("%-80s %8lld %8lld %8.3f\n", key, (long long)stored, (long long)absent,
                bytes / 1e9);
        }
        free(folder);
    }
    if (ok) {
        struct statvfs volume;
        double free_bytes =
            statvfs(root, &volume) == 0 ? (double)volume.f_bavail * volume.f_frsize : 0.;
        printf(
            "total %.3f GB in %s; %.0f GB free on the disc\n", total / 1e9, root, free_bytes / 1e9);
    }
    sqlite3_finalize(registered);
    sqlite3_close(db);
    globfree(&levels);
    free(pattern);
    free(registry_path);
    free(root);
    return !ok;
}

static int cached_ct_is_dir(const char *folder) {
    struct stat st;
    return lstat(folder, &st) == 0 && S_ISDIR(st.st_mode);
}

static int cached_ct_level(const char *folder) {
    const char *name = strrchr(folder, '/') + 1;
    char *index = path_join(folder, "index.sqlite");
    int ok = *name && !name[strspn(name, "0123456789")] && cached_ct_is_dir(folder) && index &&
        access(index, F_OK) == 0;
    free(index);
    return ok;
}

static int cached_ct_array(const char *root, const char *name, size_t length) {
    char *file = path_join(root, "arrays.sqlite");
    sqlite3 *db = NULL;
    sqlite3_stmt *stmt = NULL;
    const char sql[] = "SELECT 1 FROM arrays WHERE sample||'/'||name=?1 LIMIT 1";
    int ok = file && cached_ct_prepare(file, &db, sql, &stmt) == SQLITE_OK &&
        sqlite3_bind_text(stmt, 1, name, (int)length, SQLITE_STATIC) == SQLITE_OK &&
        sqlite3_step(stmt) == SQLITE_ROW;
    free(file);
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return ok;
}

static int cached_ct_clear(int parent, const char *path, int64_t *freed) {
    struct stat st;
    if (fstatat(parent, path, &st, AT_SYMLINK_NOFOLLOW)) {
        return 0;
    }
    if (!S_ISDIR(st.st_mode)) {
        if (unlinkat(parent, path, 0)) {
            return 0;
        }
        if (S_ISREG(st.st_mode)) {
            *freed += st.st_size;
        }
        return 1;
    }
    int fd = openat(parent, path, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    DIR *dir = fd < 0 ? NULL : fdopendir(fd);
    if (!dir) {
        if (fd >= 0) {
            close(fd);
        }
        return 0;
    }
    int ok = 1;
    for (struct dirent *entry = readdir(dir); ok && entry; entry = readdir(dir)) {
        if (cached_ct_safe(entry->d_name)) {
            ok = cached_ct_clear(fd, entry->d_name, freed);
        }
    }
    closedir(dir);
    return ok && unlinkat(parent, path, AT_REMOVEDIR) == 0;
}

static int cached_ct_writer_lock(const char *folder) {
    char *path = path_join(folder, "writer.lock");
    int fd = path ? open(path, O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0644) : -1;
    free(path);
    if (fd >= 0 && flock(fd, LOCK_EX | LOCK_NB)) {
        close(fd);
        fd = -1;
    }
    return fd;
}

int hx_chunk_evict_main(int argc, char **argv) {
    const char *key = argc == 2 ? argv[1] : argc == 3 && !strcmp(argv[1], "evict") ? argv[2] : NULL;
    if (!key) {
        fprintf(stderr, "Usage: ./run.sh scans.chunks.fetch evict SAMPLE/NAME[/LEVEL]\n");
        return 2;
    }
    char *root = cached_ct_root();
    char *candidate = root && key[0] != '/' ? path_join(root, key) : strdup(key);
    char *target = candidate ? realpath(candidate, NULL) : NULL;
    free(candidate);
    size_t width = root ? strlen(root) : 0;
    int ok = width > 1 && target && strncmp(root, target, width) == 0 && target[width] == '/' &&
        cached_ct_is_dir(target);
    const char *name = ok ? target + width + 1 : NULL, *second = name ? strchr(name, '/') : NULL;
    const char *level = second ? strchr(second + 1, '/') : NULL;
    ok = ok && second && (!level || !strchr(level + 1, '/')) &&
        cached_ct_array(root, name, level ? (size_t)(level - name) : strlen(name));
    if (!ok) {
        fprintf(stderr, "HerculeX: %s is not a registered cache array or level\n", key);
    }
    int *locks = NULL;
    size_t count = 0, capacity = 0;
    DIR *dir = ok && !level ? opendir(target) : NULL;
    ok = ok && (level || dir);
    while (ok) {
        struct dirent *entry = level ? NULL : readdir(dir);
        if (level ? count != 0 : entry == NULL) {
            break;
        }
        if (!level && !cached_ct_safe(entry->d_name)) {
            continue;
        }
        char *folder = level ? strdup(target) : path_join(target, entry->d_name);
        int known = folder && cached_ct_level(folder);
        if (!known) {
            fprintf(stderr, "HerculeX: %s contains paths outside the chunk store format\n", target);
            ok = 0;
        } else {
            if (count == capacity) {
                capacity = capacity ? capacity * 2 : 8;
                int *next = realloc(locks, capacity * sizeof(*next));
                if (next) {
                    locks = next;
                } else {
                    ok = 0;
                }
            }
            if (ok) {
                int fd = cached_ct_writer_lock(folder);
                if (fd < 0) {
                    fprintf(stderr, "HerculeX: cannot claim cache writer lock at %s\n", folder);
                    ok = 0;
                } else {
                    locks[count++] = fd;
                }
            }
        }
        free(folder);
    }
    if (dir) {
        closedir(dir);
    }
    int64_t freed = 0;
    if (ok && cached_ct_clear(AT_FDCWD, target, &freed)) {
        printf("evicted %s (%.3f GB)\n", target, freed / 1e9);
        char *parent = strrchr(target, '/');
        while (parent && (size_t)(parent - target) > width) {
            *parent = 0;
            if (rmdir(target)) {
                break;
            }
            parent = strrchr(target, '/');
        }
    } else if (ok) {
        fprintf(stderr, "HerculeX: cannot remove %s: %s\n", target, strerror(errno));
        ok = 0;
    }
    for (size_t i = 0; i < count; i++) {
        close(locks[i]);
    }
    free(locks);
    free(target);
    free(root);
    return !ok;
}
#if defined(HX_UNITY) || defined(HX_CHUNK_RESOLVE)
#include "bucket_listing.h"

static const char ct_server[] = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com";
static const char ct_origin[] = "s3://vesuvius-challenge-open-data";

static int ct_suffix(const char *s, const char *tail) {
    size_t n = strlen(s), m = strlen(tail);
    return n >= m && !strcmp(s + n - m, tail);
}

static int ct_catalogue(const char *sample, const char *scan, char **url) {
    char *lib = local_root(library_root);
    char *file = lib ? path_join(lib, "catalog/metadata.json") : NULL;
    json_object *doc = file ? json_object_from_file(file) : NULL;
    if (!doc) {
        fprintf(stderr, "cannot read scan catalogue %s; run ./run.sh catalog first\n",
            file ? file : "");
    }
    free(file);
    free(lib);
    json_object *row = json_object_object_get(doc, "samples");
    row = json_object_object_get(row, sample);
    row = json_object_object_get(row, "volumes");
    row = json_object_object_get(row, scan);
    json_object *products = json_object_object_get(row, "data");
    int status = doc ? 1 : -1;
    for (size_t i = 0;
        json_object_is_type(products, json_type_array) && i < json_object_array_length(products);
        i++) {
        json_object *product = json_object_array_get_idx(products, i);
        const char *type = json_object_get_string(json_object_object_get(product, "type"));
        if (!type || strcmp(type, "ome-zarr")) {
            continue;
        }
        json_object *origin =
            json_object_array_get_idx(json_object_object_get(product, "origins"), 0);
        json_object *root =
            json_object_array_get_idx(json_object_object_get(origin, "access_roots"), 0);
        const char *path = json_object_get_string(json_object_object_get(origin, "path"));
        const char *base = json_object_get_string(json_object_object_get(root, "url"));
        if (path && strstr(path, ".vcd3d.")) {
            continue;
        }
        status = -1;
        if (path && base) {
            if (!strcmp(base, ct_origin)) {
                const char *mirror = getenv("HERCULEX_CT_BUCKET");
                base = mirror && *mirror ? mirror : ct_server;
            }
            path += strspn(path, "/");
            size_t n = strlen(path), m = strlen(base);
            while (n && path[n - 1] == '/') {
                n--;
            }
            while (m && base[m - 1] == '/') {
                m--;
            }
            if (n && m && asprintf(url, "%.*s/%.*s", (int)m, base, (int)n, path) >= 0) {
                status = 0;
            }
        }
        break;
    }
    json_object_put(doc);
    return status;
}

static int ct_make_folder(const char *root) {
    char *parent = strdup(root);
    if (!parent) {
        return 0;
    }
    int ok = 1;
    for (char *s = parent + 1; ok; s++) {
        char end = *s;
        if (!end || end == '/') {
            *s = 0;
            ok = mkdir(parent, 0755) == 0 || errno == EEXIST;
            *s = end;
        }
        if (!end) {
            break;
        }
    }
    free(parent);
    return ok;
}

static int ct_open_registry(const char *registry, sqlite3 **db) {
    int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE;
    int ok = sqlite3_open_v2(registry, db, flags, NULL) == SQLITE_OK;
    if (ok) {
        sqlite3_busy_timeout(*db, 60000);
        const char schema[] =
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;"
            "CREATE TABLE IF NOT EXISTS arrays (root TEXT NOT NULL, level INTEGER NOT NULL,"
            "url TEXT NOT NULL UNIQUE, sample TEXT NOT NULL, name TEXT NOT NULL, shape TEXT NOT "
            "NULL,"
            "chunks TEXT NOT NULL, dtype TEXT NOT NULL, fill_value INTEGER NOT NULL, compressor "
            "TEXT,"
            "separator TEXT NOT NULL, scale TEXT NOT NULL, added REAL NOT NULL,"
            "PRIMARY KEY(root,level), UNIQUE(sample,name,level));"
            "CREATE TABLE IF NOT EXISTS resolved (sample TEXT NOT NULL, scan TEXT NOT NULL, "
            "kind TEXT NOT NULL, url TEXT NOT NULL, level INTEGER, source TEXT NOT NULL, "
            "candidates TEXT NOT NULL, resolved REAL NOT NULL, PRIMARY KEY(sample,scan,kind));"
            "CREATE TABLE IF NOT EXISTS documents (url TEXT PRIMARY KEY, body BLOB NOT NULL,"
            "fetched REAL NOT NULL);";
        ok = sqlite3_exec(*db, schema, NULL, NULL, NULL) == SQLITE_OK;
    }
    return ok;
}

static int ct_save_resolution(
    const char *registry, const char *sample, const char *scan, const char *url, int refresh) {
    sqlite3 *db = NULL;
    sqlite3_stmt *stmt = NULL;
    const char sql[] =
        "INSERT INTO resolved VALUES(?1,?2,'ct',?3,0,'catalog',json_array(?3),unixepoch()) "
        "ON CONFLICT(sample,scan,kind) DO UPDATE SET url=excluded.url,level=0,source='catalog',"
        "candidates=excluded.candidates,resolved=excluded.resolved "
        "WHERE resolved.url<>excluded.url OR resolved.level IS NOT 0 "
        "OR resolved.source<>'catalog' OR ?4";
    int ok = ct_open_registry(registry, &db) &&
        sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK &&
        sqlite3_bind_text(stmt, 1, sample, -1, SQLITE_STATIC) == SQLITE_OK &&
        sqlite3_bind_text(stmt, 2, scan, -1, SQLITE_STATIC) == SQLITE_OK &&
        sqlite3_bind_text(stmt, 3, url, -1, SQLITE_STATIC) == SQLITE_OK &&
        sqlite3_bind_int(stmt, 4, refresh) == SQLITE_OK && sqlite3_step(stmt) == SQLITE_DONE;
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return ok;
}

int hx_chunk_resolve(const char *sample, const char *scan, int refresh, char **url, int *level) {
    *url = NULL;
    *level = 0;
    int status = ct_catalogue(sample, scan, url);
    if (!status) {
        char *root = cached_ct_root();
        char *file = root ? path_join(root, "arrays.sqlite") : NULL;
        if (!file || !ct_make_folder(root) ||
            !ct_save_resolution(file, sample, scan, *url, refresh)) {
            free(*url);
            *url = NULL;
            status = -1;
        }
        free(file);
        free(root);
    }
    return status;
}

int hx_chunk_resolve_main(int argc, char **argv) {
    const char *sample = NULL, *scan = NULL;
    int refresh = 0, ok = 1;
    const char usage[] = "Usage: ./run.sh scans.chunks.fetch resolve SAMPLE SCAN [--refresh]\n";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            fputs(usage, stdout);
            return 0;
        }
        if (!strcmp(argv[i], "--refresh")) {
            refresh = 1;
        } else if (argv[i][0] == '-' || scan) {
            ok = 0;
        } else if (!sample) {
            sample = argv[i];
        } else {
            scan = argv[i];
        }
    }
    if (!ok || !scan) {
        fputs(usage, stderr);
        return 2;
    }
    char *url;
    int level;
    int status = hx_chunk_resolve(sample, scan, refresh, &url, &level);
    if (status == 0) {
        if (level < 0) {
            printf("%-10s    %s\n", "ct", url);
        } else {
            printf("%-10s %2d %s\n", "ct", level, url);
        }
    } else if (status > 0) {
        fprintf(stderr, "no CT volume published for %s scan %s\n", sample, scan);
    } else {
        fprintf(stderr, "cannot resolve CT for %s scan %s\n", sample, scan);
    }
    free(url);
    return status != 0;
}
#endif
#ifdef HX_UNITY
#include <blosc.h>
#include <math.h>
#include <openssl/sha.h>
#include <signal.h>
#include <sys/mman.h>
#include <zlib.h>
#include <zstd.h>

typedef struct {
    char *url, *sample, *name, *separator;
    int64_t shape[3], chunks[3], grid[3], bytes;
    char *codec;
} CtArray;

static const char *ct_json_text(json_object *row, const char *field) {
    return json_object_get_string(json_object_object_get(row, field));
}

static json_object *ct_json_url(CURL *curl, const char *root, const char *leaf, int *ok) {
    char *url = path_join(root, leaf), error[128] = "";
    char *body = url ? inventory_page(curl, url, error) : NULL;
    json_object *doc = body ? json_tokener_parse(body) : NULL;
    *ok = doc != NULL || (body == NULL && strncmp(error, "HTTP Error 404:", 15) == 0);
    free(body);
    free(url);
    return doc;
}

static json_object *ct_scale(json_object *attrs, int level) {
    char level_name[32];
    snprintf(level_name, sizeof(level_name), "%d", level);
    json_object *multiscale = json_object_object_get(attrs, "multiscales");
    for (size_t i = 0; json_object_is_type(multiscale, json_type_array) &&
        i < json_object_array_length(multiscale);
        i++) {
        json_object *set =
            json_object_object_get(json_object_array_get_idx(multiscale, i), "datasets");
        for (size_t j = 0;
            json_object_is_type(set, json_type_array) && j < json_object_array_length(set); j++) {
            json_object *d = json_object_array_get_idx(set, j);
            const char *path = ct_json_text(d, "path");
            json_object *changes = json_object_object_get(d, "coordinateTransformations");
            if (!path || strcmp(path, level_name) != 0) {
                continue;
            }
            for (size_t k = 0; json_object_is_type(changes, json_type_array) &&
                k < json_object_array_length(changes);
                k++) {
                json_object *t = json_object_array_get_idx(changes, k);
                const char *type = ct_json_text(t, "type");
                json_object *scale = json_object_object_get(t, "scale");
                if (type && strcmp(type, "scale") == 0 &&
                    json_object_is_type(scale, json_type_array)) {
                    size_t n = json_object_array_length(scale);
                    json_object *copy = json_object_new_array();
                    for (size_t m = n > 3 ? n - 3 : 0; copy && m < n; m++) {
                        json_object *item = json_object_new_double(
                            json_object_get_double(json_object_array_get_idx(scale, m)));
                        if (!item || json_object_array_add(copy, item) != 0) {
                            json_object_put(item);
                            json_object_put(copy);
                            copy = NULL;
                        }
                    }
                    return copy;
                }
            }
        }
    }
    json_object *scale = json_object_new_array();
    for (int i = 0; scale && i < 3; i++) {
        json_object *item = json_object_new_double(pow(2, level));
        if (!item || json_object_array_add(scale, item) != 0) {
            json_object_put(item);
            json_object_put(scale);
            scale = NULL;
        }
    }
    return scale;
}

static int ct_array_names(const char *root, char **sample, char **name) {
    CURLU *url = curl_url();
    char *path = NULL;
    int ok = url && curl_url_set(url, CURLUPART_URL, root, 0) == CURLUE_OK &&
        curl_url_get(url, CURLUPART_PATH, &path, 0) == CURLUE_OK;
    if (ok) {
        char *first = path + strspn(path, "/"), *last = first + strlen(first);
        while (last > first && last[-1] == '/') {
            *--last = 0;
        }
        char *parent = NULL, *part = first;
        for (char *s = first; *s; s++) {
            if (*s == '/') {
                *s = 0;
                parent = part;
                part = s + 1;
            }
        }
        if (strcmp(first, "samples") == 0 && parent && parent != first) {
            first += strlen(first) + 1;
        }
        if (ct_suffix(part, ".zarr")) {
            part[strlen(part) - 5] = 0;
        }
        if (ct_suffix(part, ".ome")) {
            part[strlen(part) - 4] = 0;
        }
        *sample = strdup(first);
        *name = NULL;
        size_t length = strlen(first);
        if (parent && parent != first && !strncmp(part, first, length) && part[length] == '_' &&
            strcmp(parent, "volumes") != 0 && strcmp(parent, "surfaces") != 0 &&
            strcmp(parent, "predictions") != 0 && strcmp(parent, "representations") != 0) {
            ok = asprintf(name, "%s-%s", parent, part + length + 1) >= 0;
        } else {
            *name = strdup(part);
        }
        ok = ok && cached_ct_safe(*sample) && cached_ct_safe(*name);
    }
    curl_free(path);
    curl_url_cleanup(url);
    return ok;
}

static int ct_missing_metadata(sqlite3 *db, const char *path, int save) {
    sqlite3_stmt *stmt = NULL;
    const char *sql = save ? "INSERT OR REPLACE INTO documents VALUES(?1,X'',unixepoch())"
                           : "SELECT 1 FROM documents WHERE url=?1 AND length(body)=0 "
                             "AND fetched>unixepoch()-300";
    int ok = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK &&
        sqlite3_bind_text(stmt, 1, path, -1, SQLITE_STATIC) == SQLITE_OK &&
        sqlite3_step(stmt) == (save ? SQLITE_DONE : SQLITE_ROW);
    sqlite3_finalize(stmt);
    return ok;
}

static int ct_register_array(sqlite3 *db, const char *root, int level, sqlite3_stmt **row) {
    char path[32];
    snprintf(path, sizeof(path), "%d", level);
    char *url = path_join(root, path), *sample = NULL, *name = NULL;
    char *metadata = url ? path_join(url, ".zarray") : NULL;
    CURL *curl = metadata && !ct_missing_metadata(db, metadata, 0) ? curl_easy_init() : NULL;
    int ok = 0;
    json_object *meta = curl ? ct_json_url(curl, url, ".zarray", &ok) : NULL;
    if (!meta && ok) {
        ct_missing_metadata(db, metadata, 1);
    }
    free(metadata);
    const char *dtype = ct_json_text(meta, "dtype"), *order = ct_json_text(meta, "order");
    json_object *filters = json_object_object_get(meta, "filters");
    ok = meta && json_object_get_int(json_object_object_get(meta, "zarr_format")) == 2 && dtype &&
        (!strcmp(dtype, "|u1") || !strcmp(dtype, "<u1") || !strcmp(dtype, "u1")) &&
        (!order || !strcmp(order, "C")) &&
        (!filters ||
            (json_object_is_type(filters, json_type_array) &&
                json_object_array_length(filters) == 0));
    json_object *attrs = ok ? ct_json_url(curl, root, ".zattrs", &ok) : NULL;
    json_object *scale = ok ? ct_scale(attrs, level) : NULL;
    const char *shape = ct_json_text(meta, "shape"), *chunks = ct_json_text(meta, "chunks");
    const char *sep = ct_json_text(meta, "dimension_separator");
    json_object *codec = json_object_object_get(meta, "compressor");
    ok = ok && scale && shape && chunks && (!sep || strlen(sep) == 1) &&
        ct_array_names(root, &sample, &name);
    int64_t fill = json_object_get_int64(json_object_object_get(meta, "fill_value"));
    ok = ok && fill >= 0 && fill <= UINT8_MAX &&
        sqlite3_exec(db, "BEGIN IMMEDIATE", NULL, NULL, NULL) == SQLITE_OK;
    sqlite3_stmt *stmt = NULL;
    const char collision[] =
        "SELECT 1 FROM arrays WHERE sample=?1 AND name=?2 AND level=?3 AND root<>?4";
    if (ok) {
        ok = sqlite3_prepare_v2(db, collision, -1, &stmt, NULL) == SQLITE_OK &&
            sqlite3_bind_text(stmt, 1, sample, -1, SQLITE_STATIC) == SQLITE_OK &&
            sqlite3_bind_text(stmt, 2, name, -1, SQLITE_STATIC) == SQLITE_OK &&
            sqlite3_bind_int(stmt, 3, level) == SQLITE_OK &&
            sqlite3_bind_text(stmt, 4, root, -1, SQLITE_STATIC) == SQLITE_OK;
        int result = ok ? sqlite3_step(stmt) : SQLITE_ERROR;
        ok = result == SQLITE_ROW || result == SQLITE_DONE;
        if (result == SQLITE_ROW) {
            unsigned char digest[SHA_DIGEST_LENGTH];
            SHA1((const unsigned char *)root, strlen(root), digest);
            char *changed = NULL;
            ok = asprintf(&changed, "%s-%02x%02x%02x", name, digest[0], digest[1], digest[2]) >= 0;
            free(name);
            name = changed;
        }
        sqlite3_finalize(stmt);
        stmt = NULL;
    }
    const char insert[] =
        "INSERT OR IGNORE INTO arrays VALUES(?1,?2,?3,?4,?5,?6,?7,'uint8',?8,?9,?10,?11,"
        "(julianday('now')-2440587.5)*86400.0)";
    if (ok) {
        ok = sqlite3_prepare_v2(db, insert, -1, &stmt, NULL) == SQLITE_OK &&
            sqlite3_bind_text(stmt, 1, root, -1, SQLITE_STATIC) == SQLITE_OK &&
            sqlite3_bind_int(stmt, 2, level) == SQLITE_OK &&
            sqlite3_bind_text(stmt, 3, url, -1, SQLITE_STATIC) == SQLITE_OK &&
            sqlite3_bind_text(stmt, 4, sample, -1, SQLITE_STATIC) == SQLITE_OK &&
            sqlite3_bind_text(stmt, 5, name, -1, SQLITE_STATIC) == SQLITE_OK &&
            sqlite3_bind_text(stmt, 6, shape, -1, SQLITE_STATIC) == SQLITE_OK &&
            sqlite3_bind_text(stmt, 7, chunks, -1, SQLITE_STATIC) == SQLITE_OK &&
            sqlite3_bind_int64(stmt, 8, fill) == SQLITE_OK &&
            sqlite3_bind_text(stmt, 9, json_object_get_string(codec), -1, SQLITE_STATIC) ==
                SQLITE_OK &&
            sqlite3_bind_text(stmt, 10, sep ? sep : ".", -1, SQLITE_STATIC) == SQLITE_OK &&
            sqlite3_bind_text(stmt, 11, json_object_get_string(scale), -1, SQLITE_STATIC) ==
                SQLITE_OK &&
            sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
    }
    if (ok) {
        ok = sqlite3_exec(db, "COMMIT", NULL, NULL, NULL) == SQLITE_OK;
        sqlite3_reset(*row);
    }
    curl_easy_cleanup(curl);
    json_object_put(meta);
    json_object_put(attrs);
    json_object_put(scale);
    free(url);
    free(sample);
    free(name);
    return ok;
}

static int ct_lookup_array(CtArray *a, const char *registry, const char *root, int level) {
    sqlite3 *db = NULL;
    sqlite3_stmt *row = NULL;
    const char sql[] = "SELECT url,sample,name,shape,chunks,separator,compressor FROM arrays "
                       "WHERE root=rtrim(?1,'/') AND level=?2";
    int ok = ct_open_registry(registry, &db) &&
        sqlite3_prepare_v2(db, sql, -1, &row, NULL) == SQLITE_OK &&
        sqlite3_bind_text(row, 1, root, -1, SQLITE_STATIC) == SQLITE_OK &&
        sqlite3_bind_int(row, 2, level) == SQLITE_OK;
    int result = ok ? sqlite3_step(row) : SQLITE_ERROR;
    if (result == SQLITE_DONE && ct_register_array(db, root, level, &row)) {
        result = sqlite3_step(row);
    }
    if (result == SQLITE_ROW) {
        const char *value[7];
        for (int i = 0; i < 7; i++) {
            value[i] = (const char *)sqlite3_column_text(row, i);
        }
        ok = value[0] && cached_ct_safe(value[1]) && cached_ct_safe(value[2]) &&
            cached_ct_vector(value[3], a->shape) && cached_ct_vector(value[4], a->chunks) &&
            value[5] && (strcmp(value[5], "/") == 0 || strcmp(value[5], ".") == 0);
        if (ok) {
            a->url = strdup(value[0]);
            a->sample = strdup(value[1]);
            a->name = strdup(value[2]);
            a->separator = strdup(value[5]);
            json_object *codec = value[6] ? json_tokener_parse(value[6]) : NULL;
            const char *id = ct_json_text(codec, "id");
            a->codec = id ? strdup(id) : NULL;
            ok = (!value[6] || a->codec) && a->url && a->sample && a->name && a->separator;
            json_object_put(codec);
        }
        a->bytes = 1;
        for (int i = 0; ok && i < 3; i++) {
            a->grid[i] = (a->shape[i] - 1) / a->chunks[i] + 1;
            ok = a->bytes <= INT64_MAX / a->chunks[i];
            if (ok) {
                a->bytes *= a->chunks[i];
            }
        }
    } else {
        ok = 0;
    }
    sqlite3_finalize(row);
    sqlite3_close(db);
    return ok;
}

static void ct_array_keys(const CtArray *a, size_t slot, int64_t key[3]) {
    for (int i = 2; i >= 0; i--) {
        key[i] = slot % (uint64_t)a->grid[i];
        slot /= (uint64_t)a->grid[i];
    }
}

static uint64_t ct_point_word(const uint8_t *p, int width, int big) {
    uint64_t bits = 0;
    for (int i = 0; i < width; i++) {
        bits |= (uint64_t)p[big ? width - 1 - i : i] << (8 * i);
    }
    return bits;
}

static int ct_point_marks(const CtArray *a, const char *file, double margin, uint8_t *mark) {
    int fd = open(file, O_RDONLY | O_CLOEXEC);
    struct stat st;
    uint8_t *data = MAP_FAILED;
    if (fd >= 0 && fstat(fd, &st) == 0 && st.st_size >= 12 && (uint64_t)st.st_size <= SIZE_MAX) {
        data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    }
    if (fd >= 0) {
        close(fd);
    }
    if (data == MAP_FAILED) {
        return 0;
    }
    size_t prefix = data[6] == 1 ? 10 : 12;
    size_t header = ct_point_word(data + 8, (int)(prefix - 8), 0);
    char *h = memcmp(data, "\x93NUMPY", 6) == 0 && data[6] >= 1 && data[6] <= 3 &&
            header <= (size_t)st.st_size - prefix
        ? strndup((char *)data + prefix, header)
        : NULL;
    char type[8] = "";
    char *d = h ? strstr(h, "'descr':") : NULL, *dims = h ? strstr(h, "'shape':") : NULL;
    char *shape = dims ? strchr(dims, '(') : NULL, *end = NULL;
    errno = 0;
    int64_t rows = shape ? strtoll(shape + 1, &end, 10) : -1;
    int ok = shape && !errno && end != shape + 1 && rows >= 0 && *end == ',';
    if (ok) {
        errno = 0;
        long columns = strtol(end + 1, &end, 10);
        ok = !errno && columns == 3 && (*end == ')' || !strncmp(end, ",)", 2));
    }
    ok = ok && d && sscanf(d, "'descr': '%7[^']'", type) == 1 && strlen(type) == 3;
    int width = type[2] - '0', big = type[0] == '>';
    char kind = type[1];
    int fortran = h && strstr(h, "'fortran_order': True") != NULL;
    ok = ok && strchr("<>|=", type[0]) && strchr("1248", type[2]) && strchr("fiu", kind) &&
        (kind != 'f' || width != 1) && (uint64_t)rows <= (SIZE_MAX / sizeof(float)) / 3 &&
        (uint64_t)rows <= ((size_t)st.st_size - prefix - header) / (3 * width);
    float *points = ok && rows ? malloc((size_t)rows * 3 * sizeof(*points)) : NULL;
    for (int64_t i = 0; points && i < rows; i++) {
        for (int j = 0; j < 3; j++) {
            int64_t index = fortran ? j * rows + i : i * 3 + j;
            uint64_t raw = ct_point_word(data + prefix + header + index * width, width, big);
            float value;
            if (kind == 'f' && width == 2) {
                int exponent = ((raw >> 10) & 31) - 15;
                unsigned mantissa = raw & 1023;
                value = exponent == 16 ? (mantissa ? NAN : INFINITY)
                                       : ldexpf((float)(mantissa + (exponent > -15 ? 1024 : 0)),
                                             exponent > -15 ? exponent - 10 : -24);
                if (raw & 0x8000) {
                    value = -value;
                }
            } else if (kind == 'f' && width == 4) {
                uint32_t bits = (uint32_t)raw;
                memcpy(&value, &bits, sizeof(value));
            } else if (kind == 'f') {
                double f;
                memcpy(&f, &raw, sizeof(f));
                value = (float)f;
            } else if (kind == 'i') {
                unsigned shift = (8 - width) * 8;
                value = (float)((int64_t)(raw << shift) >> shift);
            } else {
                value = (float)raw;
            }
            points[i * 3 + j] = value;
        }
    }
    if (points) {
        lg_chunks_near_points(points, rows, margin, a->chunks, a->grid, mark);
    }
    ok = ok && (!rows || points);
    free(points);
    free(h);
    munmap(data, st.st_size);
    return ok;
}

static int ct_parse_box(const char *box, int64_t ends[6]) {
    const char separators[] = ":,:,:";
    for (int i = 0; i < 6; i++) {
        char *tail = NULL;
        errno = 0;
        ends[i] = strtoll(box, &tail, 10);
        if (tail == box || errno || *tail != separators[i]) {
            return 0;
        }
        box = tail + 1;
    }
    return 1;
}

static int ct_box_marks(const CtArray *a, const int64_t ends[6], uint8_t *mark) {
    int64_t low[3], high[3];
    for (int i = 0; i < 3; i++) {
        int64_t start = ends[2 * i], end = ends[2 * i + 1];
        if (start >= end) {
            return 0;
        }
        int64_t chunk = a->chunks[i];
        low[i] = start <= 0 ? 0 : min64(a->grid[i], start / chunk);
        high[i] = end <= 0 ? 0 : min64(a->grid[i], (end - 1) / chunk + 1);
    }
    if (low[2] < high[2]) {
        for (int64_t z = low[0]; z < high[0]; z++) {
            for (int64_t y = low[1]; y < high[1]; y++) {
                size_t index = (z * a->grid[1] + y) * a->grid[2] + low[2];
                memset(mark + index, 1, (size_t)(high[2] - low[2]));
            }
        }
    }
    return 1;
}

static int ct_disk_room(const char *folder) {
    struct statvfs st;
    if (statvfs(folder, &st) != 0) {
        return 0;
    }
    double free_bytes = (double)st.f_bavail * st.f_frsize;
    if (free_bytes < 60e9) {
        fprintf(
            stderr, "%.1f GB free on the store's disc, fetching needs 60 GB\n", free_bytes / 1e9);
        return 0;
    }
    return 1;
}

typedef struct {
    const CtArray *a;
    const char *folder;
    sqlite3 *db;
    sqlite3_stmt *add;
    int fd, failed, abort, notify;
    int64_t end, cached, fetched, absent, received, retries, connects;
    size_t *todo, next, count;
    pthread_mutex_t lock;
} CtTransfer;

static int ct_fetch_store(CtTransfer *t, const char *folder, int *lock) {
    char *index = path_join(folder, "index.sqlite");
    int dir = open(folder, O_DIRECTORY | O_RDONLY | O_CLOEXEC);
    if (dir >= 0) {
        *lock = openat(dir, "writer.lock", O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    }
    int ok = index && *lock >= 0 && flock(*lock, LOCK_EX) == 0;
    if (ok) {
        t->fd = openat(dir, "chunks.pack", O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    }
    if (dir >= 0) {
        close(dir);
    }
    ok = ok && t->fd >= 0 &&
        sqlite3_open_v2(index, &t->db, SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE, NULL) ==
            SQLITE_OK;
    free(index);
    if (!ok) {
        return 0;
    }
    sqlite3_busy_timeout(t->db, 60000);
    const char setup[] =
        "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;"
        "CREATE TABLE IF NOT EXISTS chunks(z INTEGER NOT NULL,y INTEGER NOT NULL,x INTEGER NOT "
        "NULL,"
        "offset INTEGER NOT NULL,PRIMARY KEY(z,y,x)) WITHOUT ROWID;"
        "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);";
    const char identify[] =
        "INSERT OR IGNORE INTO meta VALUES ('url',?1),('chunk_bytes',CAST(?2 AS TEXT))";
    sqlite3_stmt *stmt = NULL;
    ok = sqlite3_exec(t->db, setup, NULL, NULL, NULL) == SQLITE_OK &&
        sqlite3_prepare_v2(t->db, identify, -1, &stmt, NULL) == SQLITE_OK &&
        sqlite3_bind_text(stmt, 1, t->a->url, -1, SQLITE_STATIC) == SQLITE_OK &&
        sqlite3_bind_int64(stmt, 2, t->a->bytes) == SQLITE_OK && sqlite3_step(stmt) == SQLITE_DONE;
    sqlite3_finalize(stmt);
    stmt = NULL;
    const char match[] = "SELECT (SELECT value FROM meta WHERE key='url'),"
                         "(SELECT value FROM meta WHERE key='chunk_bytes')";
    ok = ok && sqlite3_prepare_v2(t->db, match, -1, &stmt, NULL) == SQLITE_OK &&
        sqlite3_step(stmt) == SQLITE_ROW;
    const char *old = ok ? (const char *)sqlite3_column_text(stmt, 0) : NULL;
    ok = old && !strcmp(old, t->a->url) && sqlite3_column_int64(stmt, 1) == t->a->bytes;
    sqlite3_finalize(stmt);
    struct stat st;
    stmt = NULL;
    const char repair[] = "DELETE FROM chunks WHERE offset>=0 AND offset>?1";
    ok = ok && fstat(t->fd, &st) == 0 &&
        sqlite3_prepare_v2(t->db, repair, -1, &stmt, NULL) == SQLITE_OK &&
        sqlite3_bind_int64(stmt, 1, st.st_size - t->a->bytes) == SQLITE_OK &&
        sqlite3_step(stmt) == SQLITE_DONE;
    sqlite3_finalize(stmt);
    stmt = NULL;
    ok = ok &&
        sqlite3_prepare_v2(t->db, "SELECT MAX(offset) FROM chunks WHERE offset>=0", -1, &stmt,
            NULL) == SQLITE_OK &&
        sqlite3_step(stmt) == SQLITE_ROW;
    if (ok && sqlite3_column_type(stmt, 0) != SQLITE_NULL) {
        t->end = sqlite3_column_int64(stmt, 0) + t->a->bytes;
    }
    sqlite3_finalize(stmt);
    return ok && (st.st_size <= t->end || ftruncate(t->fd, t->end) == 0) &&
        sqlite3_prepare_v2(t->db, "INSERT INTO chunks VALUES(?1,?2,?3,?4)", -1, &t->add, NULL) ==
        SQLITE_OK;
}

static int ct_skip_known(CtTransfer *t) {
    sqlite3_stmt *q = NULL;
    const char sql[] = "SELECT 1 FROM chunks WHERE z=?1 AND y=?2 AND x=?3";
    int ok = sqlite3_prepare_v2(t->db, sql, -1, &q, NULL) == SQLITE_OK;
    size_t found = 0;
    for (size_t i = 0; ok && i < t->count; i++) {
        int64_t key[3];
        ct_array_keys(t->a, t->todo[i], key);
        for (int j = 0; ok && j < 3; j++) {
            ok = sqlite3_bind_int64(q, j + 1, key[j]) == SQLITE_OK;
        }
        int step = ok ? sqlite3_step(q) : SQLITE_ERROR;
        if (step == SQLITE_DONE) {
            t->todo[found++] = t->todo[i];
        }
        ok = (step == SQLITE_ROW || step == SQLITE_DONE) && sqlite3_reset(q) == SQLITE_OK;
    }
    sqlite3_finalize(q);
    t->cached = t->count - found;
    t->count = found;
    return ok;
}

typedef struct {
    uint8_t *data;
    size_t used, cap;
} CtBody;

static volatile sig_atomic_t ct_interrupted;

static void ct_interrupt(int signal) {
    (void)signal;
    ct_interrupted = 1;
}

static int ct_progress(void *context, curl_off_t a, curl_off_t b, curl_off_t c, curl_off_t d) {
    CtTransfer *t = context;
    (void)a;
    (void)b;
    (void)c;
    (void)d;
    return t->notify && ct_interrupted;
}

static size_t ct_receive(void *data, size_t size, size_t count, void *context) {
    CtBody *b = context;
    if (size && count > (b->cap - b->used) / size) {
        return 0;
    }
    size_t received = size * count;
    memcpy(b->data + b->used, data, received);
    b->used += received;
    return received;
}

static int ct_decode(const CtArray *a, const CtBody *b, uint8_t *pixels) {
    size_t n = (size_t)a->bytes;
    if (strcmp(a->codec, "blosc") == 0) {
        size_t size;
        return blosc_cbuffer_validate(b->data, b->used, &size) == 0 && size == n &&
            blosc_decompress_ctx(b->data, pixels, n, 1) == a->bytes;
    }
    if (strcmp(a->codec, "zstd") == 0) {
        return ZSTD_decompress(pixels, n, b->data, b->used) == n;
    }
    if (strcmp(a->codec, "zlib") == 0 || strcmp(a->codec, "gzip") == 0) {
        if (b->used > UINT_MAX || n > UINT_MAX) {
            return 0;
        }
        z_stream z = {.next_in = b->data,
            .avail_in = (unsigned)b->used,
            .next_out = pixels,
            .avail_out = (unsigned)n};
        if (inflateInit2(&z, 15 + 32) != Z_OK) {
            return 0;
        }
        int ok = inflate(&z, Z_FINISH) == Z_STREAM_END && z.total_out == n;
        inflateEnd(&z);
        return ok;
    }
    return 0;
}

static int ct_write_chunk(int fd, const uint8_t *pixels, size_t bytes, int64_t offset) {
    while (bytes) {
        ssize_t written = pwrite(fd, pixels, bytes, offset);
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written <= 0) {
            return 0;
        }
        pixels += written;
        bytes -= (size_t)written;
        offset += written;
    }
    return 1;
}

static int ct_commit_chunk(CtTransfer *t, const int64_t key[3], const uint8_t *pixels) {
    int64_t bytes = pixels ? t->a->bytes : 0;
    if (bytes &&
        (t->end > INT64_MAX - bytes || !ct_write_chunk(t->fd, pixels, (size_t)bytes, t->end))) {
        ftruncate(t->fd, t->end);
        return 0;
    }
    int ok = sqlite3_bind_int64(t->add, 1, key[0]) == SQLITE_OK &&
        sqlite3_bind_int64(t->add, 2, key[1]) == SQLITE_OK &&
        sqlite3_bind_int64(t->add, 3, key[2]) == SQLITE_OK &&
        sqlite3_bind_int64(t->add, 4, pixels ? t->end : -1) == SQLITE_OK &&
        sqlite3_step(t->add) == SQLITE_DONE;
    sqlite3_reset(t->add);
    if (ok) {
        t->end += bytes;
    } else if (bytes) {
        ftruncate(t->fd, t->end);
    }
    return ok;
}

static void *ct_fetch_worker(void *context) {
    CtTransfer *t = context;
    const CtArray *a = t->a;
    CURL *curl = curl_easy_init();
    CtBody body = {.cap = (size_t)a->bytes * 2 + (1 << 16)};
    body.data = malloc(body.cap);
    uint8_t *decoded = a->codec ? malloc((size_t)a->bytes) : NULL;
    if (!curl || !body.data || (a->codec && !decoded)) {
        pthread_mutex_lock(&t->lock);
        t->abort = 1;
        pthread_mutex_unlock(&t->lock);
        goto finish;
    }
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 15L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, ct_receive);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, ct_progress);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, t);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    for (;;) {
        pthread_mutex_lock(&t->lock);
        int run = !t->abort && !(t->notify && ct_interrupted) && t->next < t->count;
        size_t slot = run ? t->todo[t->next++] : 0;
        pthread_mutex_unlock(&t->lock);
        if (!run) {
            break;
        }
        int64_t key[3];
        ct_array_keys(a, slot, key);
        char *url = NULL, sep = a->separator[0];
        if (asprintf(&url, "%s/%lld%c%lld%c%lld", a->url, (long long)key[0], sep, (long long)key[1],
                sep, (long long)key[2]) < 0) {
            pthread_mutex_lock(&t->lock);
            t->abort = 1;
            pthread_mutex_unlock(&t->lock);
            break;
        }
        curl_easy_setopt(curl, CURLOPT_URL, url);
        long status = 0, connects = 0;
        int retries = 0;
        CURLcode result = CURLE_FAILED_INIT;
        for (int i = 0; i < 5 && !(t->notify && ct_interrupted); i++) {
            body.used = 0;
            result = curl_easy_perform(curl);
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
            long new_connections = 0;
            curl_easy_getinfo(curl, CURLINFO_NUM_CONNECTS, &new_connections);
            connects += new_connections;
            if (i == 4 || (result == CURLE_OK && status != 429 && status < 500) ||
                result == CURLE_WRITE_ERROR || (t->notify && ct_interrupted)) {
                break;
            }
            retries++;
            usleep(50000u << i);
        }
        free(url);
        const uint8_t *pixels = NULL;
        int absent = result == CURLE_OK && status == 404;
        if (result == CURLE_OK && status == 200) {
            if (a->codec) {
                pixels = ct_decode(a, &body, decoded) ? decoded : NULL;
            } else if (body.used == (size_t)a->bytes) {
                pixels = body.data;
            }
        }
        pthread_mutex_lock(&t->lock);
        if (!t->abort && !(t->notify && ct_interrupted)) {
            t->received += (int64_t)body.used;
            t->retries += retries;
            t->connects += connects;
            if (pixels || absent) {
                if (ct_commit_chunk(t, key, pixels)) {
                    t->fetched += pixels != NULL;
                    t->absent += absent;
                } else {
                    t->abort = 1;
                }
                if (pixels && !t->abort && !ct_disk_room(t->folder)) {
                    t->abort = 1;
                }
            } else {
                t->failed++;
            }
        }
        pthread_mutex_unlock(&t->lock);
    }
finish:
    curl_easy_cleanup(curl);
    free(body.data);
    free(decoded);
    return NULL;
}

static int ct_population(CtArray *a, const char *folder, const int64_t *box, const char *points,
    double margin, int connections, int notify) {
    size_t grid = 1;
    int ok = a->bytes > 0 && (uint64_t)a->bytes <= (SIZE_MAX - (1 << 16)) / 3;
    for (int i = 0; ok && i < 3; i++) {
        ok = a->grid[i] <= INT64_MAX / a->chunks[i] && (uint64_t)a->grid[i] <= INT64_MAX / grid;
        if (ok) {
            grid *= (size_t)a->grid[i];
        }
    }
    uint8_t *mark = ok ? calloc(grid, 1) : NULL;
    ok = mark && (box ? ct_box_marks(a, box, mark) : ct_point_marks(a, points, margin, mark));
    CtTransfer t = {
        .a = a, .folder = folder, .fd = -1, .notify = notify, .lock = PTHREAD_MUTEX_INITIALIZER};
    for (size_t i = 0; ok && i < grid; i++) {
        t.count += mark[i];
    }
    size_t requested = t.count;
    t.todo =
        t.count && t.count <= SIZE_MAX / sizeof(*t.todo) ? malloc(t.count * sizeof(*t.todo)) : NULL;
    ok = ok && (!t.count || t.todo) && ct_make_folder(folder);
    size_t n = 0;
    for (size_t i = 0; ok && t.todo && i < grid; i++) {
        if (mark[i]) {
            t.todo[n++] = i;
        }
    }
    free(mark);
    int lock = -1;
    if (ok && requested) {
        ok = ct_disk_room(folder) && ct_fetch_store(&t, folder, &lock) && ct_skip_known(&t);
    }
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    int threads = 0;
    if (ok && t.count && curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
        ok = 0;
    }
    if (ok && t.count) {
        int limit = connections ? connections : a->bytes >= (1 << 20) ? 64 : 256;
        uint64_t per_thread = (uint64_t)a->bytes * (a->codec ? 3 : 2) + (1 << 16);
        uint64_t fit = (256u << 20) / per_thread;
        limit = (int)min64(limit, fit ? (int64_t)fit : 1);
        limit = (int)min64(limit, (int64_t)t.count);
        pthread_t *ids = malloc((size_t)limit * sizeof(*ids));
        struct sigaction sig = {.sa_handler = ct_interrupt}, previous;
        sigemptyset(&sig.sa_mask);
        if (notify) {
            ct_interrupted = 0;
        }
        int installed = notify && sigaction(SIGINT, &sig, &previous) == 0;
        if (ids) {
            for (; threads < limit; threads++) {
                if (pthread_create(&ids[threads], NULL, ct_fetch_worker, &t)) {
                    break;
                }
            }
        }
        for (int i = 0; i < threads; i++) {
            pthread_join(ids[i], NULL);
        }
        if (installed) {
            sigaction(SIGINT, &previous, NULL);
        }
        free(ids);
        curl_global_cleanup();
        ok = threads > 0;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double seconds = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) * 1e-9;
    if (ok && notify) {
        double rate = (t.fetched + t.absent + t.failed) / fmax(seconds, 1e-9);
        printf("%zu chunks: %lld cached, %lld fetched, %lld absent, %d failed; "
               "%.1f MB received in %.1f s (%.0f chunks/s, %.1f MB/s, "
               "%d connections, %lld connects, %lld retries)%s\n",
            requested, (long long)t.cached, (long long)t.fetched, (long long)t.absent, t.failed,
            t.received / 1e6, seconds, rate, t.received / 1e6 / fmax(seconds, 1e-9), threads,
            (long long)t.connects, (long long)t.retries, t.abort ? " ABORTED" : "");
    }
    sqlite3_finalize(t.add);
    sqlite3_close(t.db);
    if (t.fd >= 0) {
        close(t.fd);
    }
    if (lock >= 0) {
        close(lock);
    }
    int interrupt = notify && ct_interrupted;
    if (notify) {
        ct_interrupted = 0;
    }
    pthread_mutex_destroy(&t.lock);
    free(t.todo);
    if (interrupt) {
        fprintf(
            stderr, "interrupted; what was stored is kept, run the same command again to resume\n");
    }
    return interrupt ? 130 : !ok || t.abort || t.failed;
}

static int ct_fetch_url(const char *url, int level, const int64_t *box, const char *points,
    double margin, int connections, int notify, int64_t *shape) {
    char *root = NULL, *reg = NULL, *folder = NULL;
    int ok;
    CtArray a = {0};
    if (level >= 0) {
        root = cached_ct_root();
        reg = root ? path_join(root, "arrays.sqlite") : NULL;
        ok = reg && ct_make_folder(root) && ct_lookup_array(&a, reg, url, level) &&
            asprintf(&folder, "%s/%s/%s/%d", root, a.sample, a.name, level) >= 0;
    } else {
        ok = 0;
    }
    int result = 1;
    if (ok) {
        if (notify) {
            printf("Volume(%s/%s/%d shape=(%lld, %lld, %lld) chunks=(%lld, %lld, %lld)) %s\n",
                a.sample, a.name, level, (long long)a.shape[0], (long long)a.shape[1],
                (long long)a.shape[2], (long long)a.chunks[0], (long long)a.chunks[1],
                (long long)a.chunks[2], a.url);
            fflush(stdout);
        }
        if (shape) {
            memcpy(shape, a.shape, sizeof(a.shape));
            result = 0;
        } else {
            result = ct_population(&a, folder, box, points, margin, connections, notify);
        }
    }
    free(a.url);
    free(a.sample);
    free(a.name);
    free(a.separator);
    free(a.codec);
    free(folder);
    free(reg);
    free(root);
    return result;
}

static int ct_fetch(const char *sample, const char *scan, int override, const int64_t *box,
    const char *points, double margin, int connections, int notify) {
    char *url = NULL;
    int level;
    int status = hx_chunk_resolve(sample, scan, 0, &url, &level);
    if (!status) {
        status = ct_fetch_url(
            url, override >= 0 ? override : level, box, points, margin, connections, notify, NULL);
    }
    if (status && notify) {
        fprintf(stderr, "cannot fetch CT for %s scan %s\n", sample, scan);
    }
    free(url);
    return status;
}

int hx_chunk_url_shape(const char *url, int level, int64_t shape[3]) {
    return ct_fetch_url(url, level, NULL, NULL, 0, 0, 0, shape);
}

int hx_chunk_fetch_url_box(const char *url, int level, const int64_t box[6], int connections) {
    return ct_fetch_url(url, level, box, NULL, 0, connections, 0, NULL);
}

int hx_chunk_fetch_box(
    const char *sample, const char *scan, int level, const int64_t box[6], int connections) {
    return ct_fetch(sample, scan, level, box, NULL, 0, connections, 0);
}

int hx_chunk_fetch_main(int argc, char **argv) {
    const char *keys[] = {"--level", "--box", "--points", "--margin", "--connections"};
    const char *opts[5] = {0}, *args[3] = {0};
    int used = 0, help = argc == 1, ok = 1;
    const char usage[] =
        "Usage: ./run.sh scans.chunks.fetch SAMPLE SCAN ct [--level L] "
        "(--box z0:z1,y0:y1,x0:x1 | --points FILE.npy [--margin M]) [--connections N]\n";
    for (int i = 1; ok && i < argc; i++) {
        const char *x = argv[i];
        if (!strcmp(x, "--help") || !strcmp(x, "-h")) {
            help = 1;
            break;
        }
        if (*x != '-' && used < 3) {
            args[used++] = x;
            continue;
        }
        const char *equal = strchr(x, '=');
        size_t width = equal ? (size_t)(equal - x) : strlen(x);
        int j = 0;
        while (j < 5 && (strlen(keys[j]) != width || strncmp(keys[j], x, width) != 0)) {
            j++;
        }
        if (j == 5 || (!equal && i + 1 == argc)) {
            ok = 0;
        } else {
            opts[j] = equal ? equal + 1 : argv[++i];
        }
    }
    if (help) {
        fputs(usage, stdout);
        return 0;
    }
    int number[2] = {-1, 0};
    for (int i = 0; ok && i < 2; i++) {
        const char *value = opts[4 * i];
        if (value) {
            char *tail;
            errno = 0;
            long v = strtol(value, &tail, 10);
            ok = !errno && tail != value && !*tail && v >= 0 && v <= INT_MAX;
            if (ok) {
                number[i] = (int)v;
            }
        }
    }
    char *end = NULL;
    double margin = opts[3] ? strtod(opts[3], &end) : 2.0;
    ok = ok && (!end || (end != opts[3] && !*end && isfinite(margin)));
    if (!ok || used != 3 || strcmp(args[2], "ct") != 0 || !!opts[1] == !!opts[2]) {
        fputs(usage, stderr);
        return 2;
    }
    int64_t box[6];
    if (opts[1] && !ct_parse_box(opts[1], box)) {
        fputs(usage, stderr);
        return 2;
    }
    return ct_fetch(
        args[0], args[1], number[0], opts[1] ? box : NULL, opts[2], margin, number[1], 1);
}

#endif
