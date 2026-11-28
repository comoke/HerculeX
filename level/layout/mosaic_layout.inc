#include "packing.h"

enum { BLOCK_EDGES = 1000000 };

typedef struct {
    size_t unit, source_slot;
    int64_t id, source, level;
    int height, width, stride, rows_up, outward, turn, flip_rows, flip_cols;
    int block[2], overlap[4];
    const char *folder;
    double um, side, zlo, zhi, radius, area, span;
    uint8_t *mask;
} PlacedPiece;

typedef struct {
    size_t unit, first, facing;
    int64_t source, level;
    double width, span, area, place, along;
} LayoutSource;

typedef struct {
    HxLayoutJoin join;
    double row[2], column[2];
} LayoutJoint;

typedef struct {
    size_t rank, index;
    double value, weight;
} LayoutValue;

static void *layout_array(size_t count, size_t size) {
    void *data = calloc(count ? count : 1, size);
    if (!data) {
        fputs("HerculeX: out of memory laying out the sheet\n", stderr);
        exit(1);
    }
    return data;
}

static int layout_value_order(const void *one, const void *two) {
    const LayoutValue *a = one, *b = two;
    if (a->rank != b->rank) {
        return (a->rank > b->rank) - (a->rank < b->rank);
    }
    if (a->value != b->value) {
        return (a->value > b->value) - (a->value < b->value);
    }
    return (a->index > b->index) - (a->index < b->index);
}

static double layout_sum(const double *values, size_t count) {
    if (count < 8) {
        double total = -0.;
        for (size_t i = 0; i < count; i++) {
            total += values[i];
        }
        return total + 0.;
    }
    if (count <= 128) {
        double partial[8];
        memcpy(partial, values, sizeof(partial));
        size_t end = count - count % 8;
        for (size_t i = 8; i < end; i += 8) {
            for (int j = 0; j < 8; j++) {
                partial[j] += values[i + j];
            }
        }
        double total = ((partial[0] + partial[1]) + (partial[2] + partial[3])) +
            ((partial[4] + partial[5]) + (partial[6] + partial[7]));
        for (size_t i = end; i < count; i++) {
            total += values[i];
        }
        return total;
    }
    size_t middle = count / 2;
    middle -= middle % 8;
    return layout_sum(values, middle) + layout_sum(values + middle, count - middle);
}

static double layout_median(double *values, size_t count) {
    qsort(values, count, sizeof(*values), double_before);
    return (values[(count - 1) / 2] + values[count / 2]) / 2.;
}

static double layout_axis_at(const HxAxisKnot *axis, size_t count, double z, int dimension) {
    size_t lo = 0, hi = count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (axis[mid].z <= z) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    size_t b = lo < count ? lo : count - 1, a = lo ? lo - 1 : 0;
    double from = dimension ? axis[a].y : axis[a].x;
    double to = dimension ? axis[b].y : axis[b].x;
    return a == b ? from : (to - from) / (axis[b].z - axis[a].z) * (z - axis[a].z) + from;
}

static int summarize_piece(PlacedPiece *p, HxLayoutPiece *place, const Field *axis, int framed) {
    uint32_t width, height;
    int dtype;
    double *mesh = read_raw_mesh(p->folder, &width, &height, &dtype);
    json_object *meta = read_record(p->folder, "meta.json");
    p->um = json_object_get_double(json_object_object_get(meta, "voxel_um")) / 1000.;
    p->stride = json_object_get_int(json_object_object_get(meta, "stride"));
    json_object_put(meta);
    if (!mesh || !isfinite(p->um) || p->um <= 0 || p->stride <= 0 || height > INT_MAX - 4 ||
        width > INT_MAX - 4) {
        free(mesh);
        return 0;
    }
    p->width = width;
    p->height = height;
    p->side = p->um * p->stride;
    size_t cells = (size_t)height * width, known = 0;
    const double *x = mesh, *y = x + cells, *z = y + cells;
    p->zlo = INFINITY;
    p->zhi = -INFINITY;
    for (size_t i = 0; i < cells; i++) {
        if (x[i] > 0) {
            known++;
            p->zlo = fmin(p->zlo, z[i]);
            p->zhi = fmax(p->zhi, z[i]);
        }
    }
    if (!known) {
        free(mesh);
        return 0;
    }
    double *values = layout_array(known, sizeof(*values));
    size_t step = known >= 4000 ? known / 4000 : 1, seen = 0, count = 0;
    double sums[3][3] = {{0}}, target[3] = {0}, solution[3] = {0};
    for (size_t i = 0; i < cells; i++) {
        if (x[i] <= 0 || seen++ % step) {
            continue;
        }
        size_t row = i / width, col = i % width;
        double sample[3] = {row, col, 1};
        for (int a = 0; a < 3; a++) {
            target[a] += sample[a] * z[i];
            for (int b = 0; b < 3; b++) {
                sums[a][b] += sample[a] * sample[b];
            }
        }
        values[count++] = hypot(x[i] - layout_axis_at(axis->axis, axis->axis_count, z[i], 0),
            y[i] - layout_axis_at(axis->axis, axis->axis_count, z[i], 1));
    }
    solve_three(sums, target, solution);
    p->rows_up = solution[0] > 0;
    p->radius = layout_median(values, count) * p->um;
    if (dtype == 32) {
        p->zlo = (float)p->zlo * (float)p->um;
        p->zhi = (float)p->zhi * (float)p->um;
    } else {
        p->zlo *= p->um;
        p->zhi *= p->um;
    }
    count = 0;
    for (size_t i = 0; i < cells; i++) {
        if (x[i] > 0) {
            values[count++] = z[i];
        }
    }
    double middle = layout_median(values, count);
    if (dtype == 32) {
        middle = (float)middle;
    }
    free(values);
    double cx = layout_axis_at(axis->axis, axis->axis_count, middle, 0);
    double cy = layout_axis_at(axis->axis, axis->axis_count, middle, 1);
    p->outward = hx_layout_outward(mesh, height, width, cx, cy);
    int rows = (int)(height * p->side) + 1, cols = (int)(width * p->side) + 1;
    p->mask = layout_array((size_t)rows * cols, 1);
    double reciprocal = 1. / p->side;
    for (int r = 0; r < rows; r++) {
        int row = fmin((int)(r / p->side), height - 1);
        int trim_row = fmin((int)(r * reciprocal), height - 1);
        for (int c = 0; c < cols; c++) {
            int col = fmin((int)(c / p->side), width - 1);
            int trim_col = fmin((int)(c * reciprocal), width - 1);
            p->mask[(size_t)r * cols + c] = x[(size_t)row * width + col] > 0 &&
                (!framed ||
                    (trim_row >= p->overlap[0] && trim_row < (int)height - p->overlap[1] &&
                        trim_col >= p->overlap[2] && trim_col < (int)width - p->overlap[3]));
        }
    }
    *place = (HxLayoutPiece){.mask = p->mask,
        .row_stride = cols,
        .column_stride = 1,
        .rows = rows,
        .columns = cols,
        .area = p->area};
    free(mesh);
    return 1;
}

static void piece_flips(PlacedPiece *p, int curl) {
    hx_layout_flips(p->rows_up, p->outward, curl, &p->flip_rows, &p->flip_cols);
}

static void orient_mask(PlacedPiece *p, HxLayoutPiece *place) {
    if (p->flip_rows) {
        place->mask += (place->rows - 1) * place->row_stride;
        place->row_stride = -place->row_stride;
    }
    if (p->flip_cols) {
        place->mask += place->columns - 1;
        place->column_stride = -1;
    }
}

static size_t layout_piece_at(const PlacedPiece *pieces, size_t count, int unit, int64_t id) {
    for (size_t i = 0; i < count; i++) {
        if (pieces[i].unit == (size_t)unit && pieces[i].id == id) {
            return i;
        }
    }
    return SIZE_MAX;
}

static double joint_x(const PlacedPiece *p, double column) {
    return (p->flip_cols ? p->width - 1 - column : column) * p->side;
}

static double joint_y(const PlacedPiece *p, double row) {
    return (p->flip_rows ? p->height - 1 - row : row) * p->side;
}

static double interpolate_layout(const double *x, const double *y, size_t length, double value) {
    size_t lo = 0, hi = length;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (x[mid] <= value) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (!lo || lo == length) {
        return y[lo ? lo - 1 : 0];
    }
    return (y[lo] - y[lo - 1]) / (x[lo] - x[lo - 1]) * (value - x[lo - 1]) + y[lo - 1];
}

static int layout_arc(LayoutSource *sources, size_t count, double *range, double *roll) {
    double low = INFINITY, high = -INFINITY;
    for (size_t i = 0; i < count; i++) {
        low = fmin(low, sources[i].place);
        high = fmax(high, sources[i].place);
    }
    range[0] = low;
    range[1] = high;
    double first = floor(low) - .25, length = (ceil(high) + .5 - first) * 4.;
    if (!(length > 0 && length < INT_MAX)) {
        return 0;
    }
    size_t bins = (size_t)length;
    double *data = layout_array(bins * 5, sizeof(*data));
    double *edge = data, *known = edge + bins, *measured = known + bins;
    double *filled = measured + bins, *arc = filled + bins;
    LayoutValue *values = layout_array(count, sizeof(*values));
    size_t used = 0;
    for (size_t i = 0; i < count; i++) {
        double span = fabs(sources[i].span);
        if (span >= .1) {
            values[used++] = (LayoutValue){(size_t)((sources[i].place - first) * 4.), i,
                sources[i].width / span, sources[i].area};
        }
    }
    if (!used) {
        free(values);
        free(data);
        return 0;
    }
    qsort(values, used, sizeof(*values), layout_value_order);
    size_t represented = 0;
    for (size_t i = 0, j; i < used; i = j) {
        double area = 0;
        for (j = i; j < used && values[j].rank == values[i].rank; j++) {
            area += values[j].weight;
        }
        double accumulated = 0;
        size_t middle = i;
        while (middle + 1 < j && (accumulated += values[middle].weight) < area / 2) {
            middle++;
        }
        known[represented] = values[i].rank;
        measured[represented++] = values[middle].value;
    }
    for (size_t i = 0; i < bins; i++) {
        edge[i] = first + i * .25;
        filled[i] = interpolate_layout(known, measured, represented, i);
    }
    for (size_t i = 1; i < bins; i++) {
        double sum = 0;
        for (int j = -4; j <= 4; j++) {
            long at = (long)i - 1 + j;
            size_t clamped = at < 0 ? 0 : (size_t)at < bins ? (size_t)at : bins - 1;
            sum += filled[clamped] * (1. / 9.);
        }
        arc[i] = arc[i - 1] + sum * .25;
    }
    *roll = arc[bins - 1];
    for (size_t i = 0; i < count; i++) {
        sources[i].along = *roll - interpolate_layout(edge, arc, bins, sources[i].place);
    }
    free(values);
    free(data);
    return 1;
}

static void layout_record_piece(json_object *items, const PlacedPiece *p,
    const HxLayoutPiece *place, const char *unit, int framed, int band) {
    json_object *row = json_object_new_object();
    json_object_object_add(row, "unit", json_object_new_string(unit));
    json_object_object_add(row, "piece", json_object_new_int64(p->id));
    if (!band) {
        json_object_object_add(row, "turn", json_object_new_int(p->turn));
    }
    if (framed) {
        json_object_object_add(row, "source", json_object_new_int64(p->source));
        json_object_object_add(row, "level", json_object_new_int64(p->level));
        json_object *overlap = json_object_new_array_ext(4);
        for (int i = 0; i < 4; i++) {
            json_object_array_add(overlap, json_object_new_int(p->overlap[i]));
        }
        json_object_object_add(row, "overlap", overlap);
    }
    json_object_object_add(row, "area", json_object_new_double(p->area));
    json_object_object_add(row, "group", json_object_new_int(place->group));
    json_object_object_add(row, "group_area", json_object_new_double(place->group_area));
    json_object_object_add(row, "width_mm", json_object_new_double(p->width * p->side));
    json_object_object_add(row, "height_mm", json_object_new_double(p->height * p->side));
    json_object_object_add(row, "flip_rows", json_object_new_boolean(p->flip_rows));
    json_object_object_add(row, "flip_cols", json_object_new_boolean(p->flip_cols));
    json_object_object_add(row, "um", json_object_new_double(p->um));
    json_object *shape = json_object_new_array_ext(2);
    json_object_array_add(shape, json_object_new_int(p->height));
    json_object_array_add(shape, json_object_new_int(p->width));
    json_object_object_add(row, "shape", shape);
    json_object_object_add(row, "stride", json_object_new_int(p->stride));
    if (place->shifted) {
        json_object_object_add(row, "shifted_mm", json_object_new_double(place->shift));
    }
    json_object_object_add(row, "x_mm", json_object_new_double(place->x));
    json_object_object_add(row, "y_mm", json_object_new_double(place->y));
    json_object_object_add(row, "moved_from_joint_mm", json_object_new_double(place->from_joint));
    json_object_array_add(items, row);
}

static int layout_npz(
    const char *folder, const char *field, void **data, int64_t *length, const char *dtype) {
    char *path = path_join(folder, "levels.npz");
    NpzReader file = {0};
    int ok =
        path && !npz_open_reader(&file, path) && !npz_read(&file, field, dtype, 8, data, length);
    npz_close_reader(&file);
    free(path);
    return ok;
}

static int inner_search(Search *search, const PlacedPiece *p) {
    uint32_t w, h;
    int type;
    double *xyz = read_raw_mesh(p->folder, &w, &h, &type);
    if (!xyz) {
        return 0;
    }
    size_t size = (size_t)w * h;
    search->bound = 3. * p->stride;
    search->cells = layout_array(size, sizeof(*search->cells));
    for (size_t i = 0; i < size; i++) {
        if (xyz[i] > 0) {
            search->cells[search->count++] =
                (Cell){xyz[i], xyz[size + i], xyz[2 * size + i], (int)(i / w), (int)(i % w), 0};
        }
    }
    free(xyz);
    return index_cells(search);
}

typedef struct {
    void *values;
    int64_t count;
    int wide;
} JointVector;

static int inner_vector(NpzReader *file, const char *field, int real, JointVector *out) {
    if (!npz_read(file, field, real ? "<f4" : "<i4", 4, &out->values, &out->count)) {
        return 1;
    }
    out->wide = 1;
    return !npz_read(file, field, real ? "<f8" : "<i8", 8, &out->values, &out->count);
}

static size_t inner_nearest(const Search *search, const JointVector *column, size_t point) {
    double xyz[3];
    for (int i = 0; i < 3; i++) {
        size_t n = point * 3 + i;
        double v =
            column->wide ? ((const double *)column->values)[n] : ((const float *)column->values)[n];
        xyz[i] = MATCH_SCALE * v;
    }
    return nearest_cell(search, xyz[0], xyz[1], xyz[2]);
}

static int append_inner_joints(json_object *out, const char *sheet, const char *unit,
    const PlacedPiece *pieces, size_t count, int u) {
    char *path = NULL;
    const char *name = strrchr(unit, '/') + 1;
    if (asprintf(&path, "%s/between-%s.npz", sheet, name) < 0) {
        return 0;
    }
    if (!regular_file(path)) {
        free(path);
        return 1;
    }
    NpzReader file = {0};
    JointVector a = {0}, b = {0}, pa = {0}, pb = {0};
    LayoutValue *entries = NULL;
    double *pixels = NULL;
    int ok = !npz_open_reader(&file, path) && inner_vector(&file, "a", 0, &a) &&
        inner_vector(&file, "b", 0, &b) && inner_vector(&file, "pa", 1, &pa) &&
        inner_vector(&file, "pb", 1, &pb) && a.count == b.count && pa.count == pb.count &&
        a.count == pa.count / 3 && !(pa.count % 3);
    free(path);
    npz_close_reader(&file);
    if (!ok) {
        goto done;
    }

    size_t n = a.count;
    entries = layout_array(n, sizeof(*entries));
    pixels = layout_array(4 * n, sizeof(*pixels));
    for (size_t i = 0; i < n; i++) {
        int64_t x = a.wide ? ((int64_t *)a.values)[i] : ((int32_t *)a.values)[i];
        int64_t y = b.wide ? ((int64_t *)b.values)[i] : ((int32_t *)b.values)[i];
        if (x < 0 || y < 0 || x >= INT32_MAX || y >= INT32_MAX) {
            ok = 0;
            goto done;
        }
        entries[i] = (LayoutValue){
            .rank = x < y ? x : y, .value = x < y ? y : x, .index = i, .weight = x > y};
    }
    qsort(entries, n, sizeof(*entries), layout_value_order);
    for (size_t i = 0, j; i < n; i = j) {
        for (j = i + 1;
            j < n && entries[j].rank == entries[i].rank && entries[j].value == entries[i].value;
            j++) {
        }
        if (j - i < 5) {
            continue;
        }
        int64_t lo = entries[i].rank, hi = (int64_t)entries[i].value;
        size_t p = layout_piece_at(pieces, count, u, lo), q = layout_piece_at(pieces, count, u, hi);
        if (p == SIZE_MAX || q == SIZE_MAX) {
            continue;
        }
        Search x = {0}, y = {0};
        ok = inner_search(&x, pieces + p) && inner_search(&y, pieces + q);
        size_t found = 0;
        for (size_t k = i; ok && k < j; k++) {
            int swapped = entries[k].weight != 0;
            size_t first = inner_nearest(&x, swapped ? &pb : &pa, entries[k].index);
            size_t second = inner_nearest(&y, swapped ? &pa : &pb, entries[k].index);
            if (first && second) {
                const Cell *c = x.cells + first - 1, *d = y.cells + second - 1;
                pixels[found] = c->row;
                pixels[n + found] = d->row;
                pixels[2 * n + found] = c->column;
                pixels[3 * n + found++] = d->column;
            }
        }
        free(x.cells);
        free(x.table);
        free(y.cells);
        free(y.table);
        if (!ok) {
            goto done;
        }
        if (found >= 5) {
            json_object *item = json_object_new_object();
            double rows[] = {layout_median(pixels, found), layout_median(pixels + n, found)};
            double cols[] = {
                layout_median(pixels + 2 * n, found), layout_median(pixels + 3 * n, found)};
            json_object_object_add(item, "a", endpoint(u, lo));
            json_object_object_add(item, "b", endpoint(u, hi));
            json_object_object_add(item, "edges", json_object_new_int64(found));
            json_object_object_add(item, "rows", number_array(rows, 2));
            json_object_object_add(item, "columns", number_array(cols, 2));
            json_object_array_add(out, item);
        }
    }
done:
    free(pixels);
    free(entries);
    free(a.values);
    free(b.values);
    free(pa.values);
    free(pb.values);
    return ok;
}

static int layout_joints(LayoutJoint *joints, size_t offset, PlacedPiece *pieces, size_t count,
    json_object *seams, int *refused, const int64_t *first, const int64_t *held_p,
    const int64_t *held_q, int64_t held) {
    size_t length = json_object_array_length(seams);
    int ok = 1;
    for (size_t i = 0; ok && i < length; i++) {
        json_object *row = json_object_array_get_idx(seams, i);
        LayoutJoint *joint = joints + offset + i;
        size_t ends[2];
        for (int d = 0; d < 2; d++) {
            json_object *end = json_object_object_get(row, d ? "b" : "a");
            int unit = json_object_get_int(json_object_array_get_idx(end, 0));
            int64_t id = json_object_get_int64(json_object_array_get_idx(end, 1));
            ends[d] = layout_piece_at(pieces, count, unit, id);
            if (ends[d] == SIZE_MAX) {
                ok = 0;
                break;
            }
            joint->row[d] = json_object_get_double(
                json_object_array_get_idx(json_object_object_get(row, "rows"), d));
            joint->column[d] = json_object_get_double(
                json_object_array_get_idx(json_object_object_get(row, "columns"), d));
        }
        if (!ok) {
            break;
        }
        joint->join = (HxLayoutJoin){.a = ends[0],
            .b = ends[1],
            .weight = json_object_get_int(json_object_object_get(row, "edges")),
            .sign = 1};
        if (first) {
            int64_t a = first[pieces[ends[0]].unit] + pieces[ends[0]].source;
            int64_t b = first[pieces[ends[1]].unit] + pieces[ends[1]].source;
            int found = 0;
            for (int64_t j = 0; !found && j < held; j++) {
                found = held_p[j] == a && held_q[j] == b;
            }
            *refused += !found;
        }
    }
    return ok;
}

static int record_write_layout(const char *path, json_object *document) {
    char *part = NULL;
    if (asprintf(&part, "%s.part.%ld", path, (long)getpid()) < 0) {
        return -1;
    }
    int failed =
        json_object_to_file_ext(part, document, JSON_C_TO_STRING_PLAIN) || rename(part, path);
    if (failed) {
        unlink(part);
    }
    free(part);
    return failed;
}

static int write_band_layout(
    const char *sample, const char *scan, const char *unit, const char *output, const char *sheet);

static int write_layout(const char *sample, const char *scan, int mode) {
    time_t began = time(NULL);
    char *base = level_file(sample, scan, ""), *unit_pattern = NULL;
    char *sheet = base ? path_join(base, "sheet") : NULL;
    char *pieces_root = base ? path_join(base, "pieces") : NULL;
    char *nodes = base ? path_join(base, "nodes") : NULL;
    char *joint_path = sheet ? path_join(sheet, "joints.json") : NULL;
    char *output = sheet ? path_join(sheet, "layout.json") : NULL;
    char *frame = pieces_root ? path_join(pieces_root, "frame") : NULL;
    glob_t units = {0};
    size_t count = 0, source_count = 0, blocks = 0;
    int status = 1, band = mode == 2, curl = 0, refused = 0;
    Field axis = {.sample = (char *)sample, .volume = (char *)scan};
    int64_t *first = NULL, *levels = NULL, *held_p = NULL, *held_q = NULL, *saved_curl = NULL;
    int64_t first_count = 0, levels_count = 0, held = 0, q_count = 0, curl_count = 0;
    json_object **records = NULL, *seams = NULL, *document = NULL;
    PlacedPiece *pieces = NULL;
    HxLayoutPiece *places = NULL;
    LayoutSource *sources = NULL;
    LayoutValue *sorted = NULL;
    LayoutJoint *joints = NULL;
    HxLayoutJoin *edges = NULL;
    size_t *ranks = NULL, *order = NULL, *start = NULL;
    double *areas = NULL, *radius = NULL, *deltas = NULL;

    if (!pieces_root || !nodes || !joint_path || !output || !frame ||
        asprintf(&unit_pattern, "%s/rows-*", pieces_root) < 0) {
        goto done;
    }
    int listed = band ? GLOB_NOMATCH : glob(unit_pattern, 0, NULL, &units);
    if (listed == GLOB_NOMATCH && mode != 1) {
        band = 1;
        free(unit_pattern);
        unit_pattern = NULL;
        if (asprintf(&unit_pattern, "%s/band-*", pieces_root) < 0) {
            goto done;
        }
        listed = glob(unit_pattern, 0, NULL, &units);
    }
    if (listed) {
        goto done;
    }
    if (band) {
        status = write_band_layout(sample, scan, units.gl_pathv[0], output, sheet);
        goto done;
    }
    if (load_axis(&axis)) {
        goto done;
    }
    size_t unit_count = units.gl_pathc;
    int framed = mode == 1 || (mode < 0 && !band && unit_count > 1);
    if (framed &&
        (!layout_npz(frame, "first", (void **)&first, &first_count, "<i8") ||
            !layout_npz(frame, "level", (void **)&levels, &levels_count, "<i8") ||
            !layout_npz(frame, "held_p", (void **)&held_p, &held, "<i8") ||
            !layout_npz(frame, "held_q", (void **)&held_q, &q_count, "<i8") ||
            !layout_npz(frame, "curl", (void **)&saved_curl, &curl_count, "<i8") ||
            curl_count != 1 || first_count <= (int64_t)unit_count || held != q_count)) {
        goto done;
    }
    records = layout_array(units.gl_pathc, sizeof(*records));
    for (size_t u = 0; u < unit_count; u++) {
        records[u] = read_record(units.gl_pathv[u], "record.json");
        json_object *rows = json_object_object_get(records[u], "pieces");
        int unit_curl = json_object_get_int(json_object_object_get(records[u], "chirality"));
        if (abs(unit_curl) != 1 || (u && curl != unit_curl) ||
            !json_object_is_type(rows, json_type_array)) {
            goto done;
        }
        curl = unit_curl;
        for (size_t j = 0; j < json_object_array_length(rows); j++) {
            count += json_object_get_boolean(
                json_object_object_get(json_object_array_get_idx(rows, j), "written"));
        }
    }
    if (!count || (framed && saved_curl[0] != curl)) {
        goto done;
    }
    pieces = layout_array(count, sizeof(*pieces));
    places = layout_array(count, sizeof(*places));
    sources = layout_array(count, sizeof(*sources));
    ranks = layout_array(count, sizeof(*ranks));
    sorted = layout_array(count, sizeof(*sorted));
    order = layout_array(count, sizeof(*order));
    start = layout_array(count + 1, sizeof(*start));
    areas = layout_array(count, sizeof(*areas));
    radius = layout_array(count, sizeof(*radius));
    deltas = layout_array(count, sizeof(*deltas));
    if (regular_file(joint_path)) {
        seams = json_object_from_file(joint_path);
    } else {
        seams = scan_joints(nodes, pieces_root, framed);
        if (!seams || !make_folders(sheet) || record_write_layout(joint_path, seams)) {
            goto done;
        }
    }
    if (!json_object_is_type(seams, json_type_array)) {
        goto done;
    }
    size_t seam_count = json_object_array_length(seams);
    joints = layout_array(count + seam_count, sizeof(*joints));
    edges = layout_array(count + seam_count, sizeof(*edges));

    double high = -INFINITY, low = INFINITY;
    for (size_t u = 0, i = 0; u < unit_count; u++) {
        const char *unit = units.gl_pathv[u];
        json_object *rows = json_object_object_get(records[u], "pieces");
        double *turns = NULL;
        int64_t turn_count = 0;
        if (framed && !layout_npz(unit, "turns", (void **)&turns, &turn_count, "<f8")) {
            goto done;
        }
        size_t source_start = source_count;
        for (size_t j = 0; j < json_object_array_length(rows); j++) {
            json_object *row = json_object_array_get_idx(rows, j);
            if (!json_object_get_boolean(json_object_object_get(row, "written"))) {
                continue;
            }
            PlacedPiece *p = pieces + i;
            p->unit = u;
            p->id = json_object_get_int64(json_object_object_get(row, "piece"));
            json_object *source = json_object_object_get(row, "source");
            p->source = framed && source ? json_object_get_int64(source) : p->id;
            p->area = json_object_get_double(json_object_object_get(row, "area_mm2"));
            p->span = json_object_get_double(json_object_object_get(row, "turns_spanned"));
            json_object *block = json_object_object_get(row, "block");
            json_object *overlap = json_object_object_get(row, "overlap");
            if (json_object_is_type(block, json_type_array)) {
                for (int d = 0; d < 2; d++)
                    p->block[d] = json_object_get_int(json_object_array_get_idx(block, d));
            }
            if (json_object_is_type(overlap, json_type_array)) {
                for (int d = 0; d < 4; d++)
                    p->overlap[d] = json_object_get_int(json_object_array_get_idx(overlap, d));
            }
            char *folder = NULL;
            if (asprintf(&folder, "%s/piece-%04lld", unit, (long long)p->id) < 0) {
                free(turns);
                goto done;
            }
            p->folder = folder;
            if (!summarize_piece(p, places + i, &axis, framed)) {
                free(turns);
                goto done;
            }
            high = fmax(high, p->zhi);
            low = fmin(low, p->zlo);
            size_t k = source_start;
            while (k < source_count && sources[k].source != p->source) {
                k++;
            }
            if (k == source_count) {
                sources[k] = (LayoutSource){
                    .unit = u, .source = p->source, .first = i, .facing = i, .span = p->span};
                if (framed) {
                    if (p->source < 0 || p->source >= turn_count || first[u] < 0 ||
                        p->source >= levels_count - first[u]) {
                        free(turns);
                        goto done;
                    }
                    sources[k].level = levels[first[u] + p->source];
                    sources[k].place = (double)sources[k].level + curl * turns[p->source];
                }
                source_count++;
            } else {
                size_t anchor = sources[k].first;
                joints[blocks++] =
                    (LayoutJoint){.join = {.a = anchor, .b = i, .weight = BLOCK_EDGES, .sign = 1},
                        .row = {p->block[0] - pieces[anchor].block[0], 0},
                        .column = {p->block[1] - pieces[anchor].block[1], 0}};
                if (p->area > pieces[sources[k].facing].area) {
                    sources[k].facing = i;
                }
            }
            sources[k].width = fmax(sources[k].width, p->block[1] * p->side + p->width * p->side);
            sources[k].area += p->area;
            p->source_slot = k;
            p->level = sources[k].level;
            sorted[i] = (LayoutValue){u, i, (double)p->id, 0};
            i++;
        }
        free(turns);
    }
    qsort(sorted, count, sizeof(*sorted), layout_value_order);
    for (size_t i = 0; i < count; i++) {
        ranks[sorted[i].index] = i;
    }
    if (!framed) {
        for (size_t u = 0; u < unit_count; u++) {
            if (!append_inner_joints(seams, sheet, units.gl_pathv[u], pieces, count, (int)u)) {
                goto done;
            }
        }
        if (seam_count != json_object_array_length(seams)) {
            seam_count = json_object_array_length(seams);
            free(joints);
            free(edges);
            joints = layout_array(count + seam_count, sizeof(*joints));
            edges = layout_array(count + seam_count, sizeof(*edges));
        }
    }
    if (!layout_joints(
            joints, blocks, pieces, count, seams, &refused, first, held_p, held_q, held)) {
        goto done;
    }
    for (size_t i = 0; i < count; i++) {
        pieces[i].turn = 1;
        piece_flips(pieces + i, curl);
    }
    for (size_t i = 0; i < count; i++) {
        PlacedPiece *p = pieces + i, *facing = pieces + sources[p->source_slot].facing;
        p->flip_rows = facing->flip_rows;
        p->flip_cols = facing->flip_cols;
        orient_mask(p, places + i);
    }
    for (size_t i = 0; i < blocks + seam_count; i++) {
        const LayoutJoint *j = joints + i;
        const PlacedPiece *a = pieces + j->join.a, *b = pieces + j->join.b;
        edges[i] = j->join;
        edges[i].sign = 1;
        edges[i].x = joint_x(a, j->column[0]) - joint_x(b, j->column[1]);
        edges[i].y = joint_y(a, j->row[0]) - joint_y(b, j->row[1]);
    }
    int groups = hx_layout_mosaics(places, count, ranks, edges, blocks + seam_count, order);
    if (groups <= 0) {
        goto done;
    }
    for (size_t i = 0; i < count; i++) {
        size_t p = order[i], group = places[p].group;
        areas[group] += pieces[p].area;
        radius[group] += pieces[p].radius * pieces[p].area;
        start[group + 1]++;
    }
    for (int g = 0; g < groups; g++) {
        start[g + 1] += start[g];
    }
    double range[2], roll = 0;
    if (framed) {
        if (!layout_arc(sources, source_count, range, &roll)) {
            goto done;
        }
        for (int g = 0; g < groups; g++) {
            for (size_t i = start[g]; i < start[g + 1]; i++) {
                size_t p = order[i];
                deltas[i] = sources[pieces[p].source_slot].along - places[p].x -
                    pieces[p].width * pieces[p].side / 2;
            }
            radius[g] = layout_median(deltas + start[g], start[g + 1] - start[g]);
        }
    } else {
        if (!(high > low)) {
            goto done;
        }
        for (int g = 0; g < groups; g++) {
            sorted[g] = (LayoutValue){0, g, radius[g] / areas[g], 0};
        }
        qsort(sorted, groups, sizeof(*sorted), layout_value_order);
        double area = 0;
        for (int i = 0; i < groups; i++) {
            size_t g = sorted[i].index;
            area += areas[g];
            radius[g] = area / (high - low) / .5;
        }
        roll = radius[sorted[groups - 1].index];
        for (int g = 0; g < groups; g++) {
            radius[g] = roll - radius[g];
        }
    }
    double first_y = INFINITY;
    int group_n = 0;
    int *original_groups = layout_array(count, sizeof(*original_groups));
    for (int g = 0; g < groups; g++) {
        ranks[g] = SIZE_MAX;
    }
    for (size_t i = 0; i < count; i++) {
        HxLayoutPiece *p = places + i;
        size_t g = p->group;
        original_groups[i] = g;
        PlacedPiece *seed = pieces + order[start[g]];
        p->group_area = areas[g];
        p->x += radius[g];
        p->y += curl > 0 ? high - seed->zhi : seed->zlo - low;
        first_y = fmin(first_y, p->y);
        if (ranks[g] == SIZE_MAX) {
            ranks[g] = group_n++;
        }
        p->group = ranks[g];
    }
    for (size_t i = 0; i < count; i++) {
        places[i].y -= first_y;
    }
    size_t kept = 0;
    int moved = hx_pack_layout(places, count, 120, &kept);
    if (moved < 0) {
        free(original_groups);
        goto done;
    }
    double width = 0;
    json_object *items = json_object_new_array_ext((int)count);
    for (size_t i = 0; i < count; i++) {
        const char *name = strrchr(units.gl_pathv[pieces[i].unit], '/') + 1;
        places[i].group = original_groups[i];
        layout_record_piece(items, pieces + i, places + i, name, framed, 0);
        width = fmax(width, places[i].x);
    }
    free(original_groups);
    json_object *record = json_object_new_object();
    json_object_object_add(record, "sample", json_object_new_string(sample));
    json_object_object_add(record, "volume", json_object_new_string(scan));
    json_object_object_add(record, "chirality", json_object_new_int(curl));
    json_object_object_add(record, "joints", json_object_new_int64(seam_count));
    json_object_object_add(record, "pieces", json_object_new_int64(framed ? source_count : count));
    json_object_object_add(record, "moved", json_object_new_int(moved));
    if (!framed) {
        json_object_object_add(record, "width_mm", json_object_new_double(width));
    }
    if (!band) {
        int large_count = 0;
        for (int g = 0; g < groups; g++) {
            if (areas[g] >= 400) {
                deltas[large_count++] = areas[g];
            }
        }
        double total = layout_sum(areas, groups), large = layout_sum(deltas, large_count);
        qsort(areas, groups, sizeof(*areas), double_before);
        json_object *top = json_object_new_array_ext(6);
        for (int n = 0; n < 6 && n < groups; n++) {
            json_object_array_add(top, json_object_new_double(areas[groups - 1 - n] / 100.));
        }
        json_object_object_add(record, "mosaics", json_object_new_int(groups));
        json_object_object_add(record, "largest_mosaics_cm2", top);
        json_object_object_add(record, framed ? "in_mosaics_of_4cm2" : "mosaic_cm2_over_4",
            json_object_new_double(large / 100.));
        json_object_object_add(record, "seconds", json_object_new_int64(time(NULL) - began));
        if (framed) {
            json_object_object_add(
                record, "joints the frame does not hold", json_object_new_int(refused));
            json_object_object_add(record, "blocks", json_object_new_int64(count));
            json_object_object_add(record, "area_cm2", json_object_new_double(total / 100.));
            json_object_object_add(record, "turns", number_array(range, 2));
            json_object_object_add(record, "roll_mm", json_object_new_double(roll));
        }
    }
    document = json_object_new_object();
    json_object_object_add(document, "record", record);
    json_object_object_add(document, "pieces", items);
    if (make_folders(sheet) && !record_write_layout(output, document)) {
        puts(json_object_to_json_string_ext(record, JSON_C_TO_STRING_PLAIN));
        status = 0;
    }
done:
    if (status) {
        fprintf(stderr, "HerculeX: could not lay out %s %s\n", sample, scan);
    }
    for (size_t i = 0; pieces && i < count; i++) {
        free((char *)pieces[i].folder);
        free(pieces[i].mask);
    }
    for (size_t u = 0; records && u < units.gl_pathc; u++) {
        json_object_put(records[u]);
    }
    free(records);
    json_object_put(seams);
    json_object_put(document);
    free(deltas);
    free(radius);
    free(areas);
    free(start);
    free(order);
    free(sorted);
    free(ranks);
    free(edges);
    free(joints);
    free(sources);
    free(places);
    free(pieces);
    free(axis.axis);
    free(saved_curl);
    free(held_q);
    free(held_p);
    free(levels);
    free(first);
    globfree(&units);
    free(unit_pattern);
    free(frame);
    free(output);
    free(joint_path);
    free(nodes);
    free(pieces_root);
    free(sheet);
    free(base);
    return status;
}

int hx_layout_main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fputs("Usage: layout SCROLL SCAN [framed|rows|band]\n", stderr);
        return 2;
    }
    int mode = -1;
    if (argc == 4) {
        if (!strcmp(argv[3], "framed")) {
            mode = 1;
        } else if (!strcmp(argv[3], "band")) {
            mode = 2;
        } else if (!strcmp(argv[3], "rows")) {
            mode = 0;
        } else {
            return 2;
        }
    }
    return write_layout(argv[1], argv[2], mode);
}

int hx_fit_centre(const double *x, const double *y, const double *ux, const double *uy,
    const double *weight, size_t size, int robust, double centre[2]);

typedef struct {
    double *mesh, anchor[2], winding, place;
} BandPiece;

typedef struct {
    double turn, length;
} BandSpeed;

static void band_nearest(const double *mesh, int h, int w, int include_zero, size_t *result) {
    int *nearest = layout_array((size_t)h * w, sizeof(*nearest));
    int *site = layout_array(w, sizeof(*site));
    double *edge = layout_array(w, sizeof(*edge));
    for (int x = 0; x < w; x++) {
        int above = -1, below = -1;
        for (int y = 0; y < h; y++) {
            size_t i = (size_t)y * w + x;
            if (mesh[i] > 0 || (include_zero && mesh[i] == 0)) {
                above = y;
            }
            nearest[i] = above;
        }
        for (int y = h - 1; y >= 0; y--) {
            size_t i = (size_t)y * w + x;
            if (mesh[i] > 0 || (include_zero && mesh[i] == 0)) {
                below = y;
            }
            if (below >= 0 && (nearest[i] < 0 || below - y < y - nearest[i])) {
                nearest[i] = below;
            }
        }
    }
    for (int y = 0; y < h; y++) {
        const int *row = nearest + (size_t)y * w;
        int count = 0;
        for (int x = 0; x < w; x++) {
            if (row[x] < 0) {
                continue;
            }
            double split = -INFINITY;
            while (count) {
                split = row_split(row, y, site[count - 1], x);
                if (split > edge[count - 1]) {
                    break;
                }
                count--;
            }
            edge[count] = count ? split : -INFINITY;
            site[count++] = x;
        }
        for (int x = 0, i = 0; x < w; x++) {
            while (i + 1 < count && edge[i + 1] < x) {
                i++;
            }
            int c = site[i];
            result[(size_t)y * w + x] = count ? (size_t)row[c] * w + c : SIZE_MAX;
        }
    }
    free(edge);
    free(site);
    free(nearest);
}

static int band_normals(const double *mesh, int h, int w, double sigma, float *normal) {
    size_t cells = (size_t)h * w;
    size_t *near = layout_array(cells, sizeof(*near));
    band_nearest(mesh, h, w, 0, near);
    if (near[0] == SIZE_MAX) {
        free(near);
        return 0;
    }
    float *work = layout_array(4 * cells, sizeof(*work));
    int radius = (int)(4 * sigma + .5);
    double *weights = layout_array(radius * 2 + 1, sizeof(*weights));
    double sum = 0, factor = -.5 / (sigma * sigma);
    for (int i = -radius; i <= radius; i++) {
        weights[i + radius] = exp(factor * (double)i * i);
        sum += weights[i + radius];
    }
    for (int i = 0; i <= radius * 2; i++) {
        weights[i] /= sum;
    }
    Blur blur = {.radius = radius, .plain = plain_taps(radius), .weights = weights};
    int ok = 1;
    for (int a = 0; ok && a < 3; a++) {
        float *plane = work + a * cells;
        for (size_t i = 0; i < cells; i++) {
            plane[i] = (float)mesh[a * cells + near[i]];
        }
        ok = gaussian_mesh(plane, work + 3 * cells, plane, h, w, &blur);
    }
    if (ok) {
        float_unit_normals(work, h, w, normal);
    }
    free(near);
    free(work);
    free(weights);
    return ok;
}

static int band_centre(
    const PlacedPiece *pieces, const BandPiece *band, size_t n, double centre[2]) {
    size_t capacity = 0;
    for (size_t p = 0; p < n; p++) {
        int h = pieces[p].height, w = pieces[p].width;
        capacity += (size_t)((h + 3) / 4) * ((w + 3) / 4);
    }
    double *fields = layout_array(capacity * 5, sizeof(*fields));
    double *x = fields, *y = x + capacity, *ux = y + capacity, *uy = ux + capacity;
    double *weight = uy + capacity;
    size_t used = 0;
    int ok = 1;
    for (size_t p = 0; ok && p < n; p++) {
        int h = pieces[p].height, w = pieces[p].width;
        size_t cells = (size_t)h * w;
        const double *mesh = band[p].mesh;
        float *normal = layout_array(3 * cells, sizeof(*normal));
        ok = band_normals(mesh, h, w, 225. / (pieces[p].side * 1000.), normal);
        for (int r = 0; ok && r < h; r += 4) {
            for (int c = 0; c < w; c += 4) {
                size_t i = (size_t)r * w + c;
                float length = hypotf(normal[i], normal[cells + i]);
                if (mesh[i] >= 0 && length > .7f) {
                    x[used] = mesh[i];
                    y[used] = mesh[cells + i];
                    ux[used] = normal[i] / length;
                    uy[used] = normal[cells + i] / length;
                    weight[used++] = 1;
                }
            }
        }
        free(normal);
    }
    ok = ok && hx_fit_centre(x, y, ux, uy, weight, used, 1, centre);
    free(fields);
    return ok;
}

static void band_unwrap(double *grid, int steps, int stride) {
    double offset = 0, previous = grid[0];
    for (int i = 1; i < steps; i++) {
        double value = grid[(size_t)i * stride], delta = value - previous;
        if (fabs(delta) > M_PI) {
            double rem = fmod(delta + M_PI, 2 * M_PI);
            if (rem < 0) {
                rem += 2 * M_PI;
            }
            offset += rem - M_PI - delta;
        }
        grid[(size_t)i * stride] = value + offset;
        previous = value;
    }
}

static size_t band_pose(
    PlacedPiece *p, BandPiece *band, const double *centre, int curl, BandSpeed *speeds) {
    int h = p->height, w = p->width;
    size_t cells = (size_t)h * w, known = 0;
    const double *mesh = band->mesh, *z = mesh + 2 * cells;
    double sums[3][3] = {{0}}, target[3] = {0}, a[3] = {0};
    double *data = layout_array(3 * cells, sizeof(*data));
    double *grid = data, *rows = grid + cells, *cols = rows + cells;
    size_t *near = layout_array(cells, sizeof(*near));
    band_nearest(mesh, h, w, 1, near);
    if (near[0] == SIZE_MAX) {
        free(near);
        free(data);
        return 0;
    }
    p->zlo = INFINITY;
    p->zhi = -INFINITY;
    for (size_t i = 0; i < cells; i++) {
        size_t j = near[i];
        grid[i] = atan2(mesh[cells + j] - centre[1], mesh[j] - centre[0]);
        if (mesh[i] < 0) {
            continue;
        }
        size_t row = i / w, col = i % w;
        double sample[] = {row, col, 1};
        for (int k = 0; k < 3; k++) {
            target[k] += sample[k] * z[i];
            for (int l = 0; l < 3; l++) {
                sums[k][l] += sample[k] * sample[l];
            }
        }
        rows[known] = row;
        cols[known++] = col;
        p->zlo = fmin(p->zlo, z[i]);
        p->zhi = fmax(p->zhi, z[i]);
    }
    free(near);
    solve_three(sums, target, a);
    double up_run = a[1] / fmax(hypot(a[0], a[1]), 1e-9) * curl;
    double up_rise = a[0] / fmax(hypot(a[0], a[1]), 1e-9) * curl;
    int outward = hx_layout_outward(mesh, h, w, centre[0], centre[1]);
    double right_run = outward ? -up_rise : up_rise;
    double right_rise = outward ? up_run : -up_run;
    p->flip_rows = up_rise > 0;
    p->flip_cols = right_run < 0;
    band->anchor[0] = layout_median(cols, known);
    band->anchor[1] = layout_median(rows, known);
    p->zlo *= p->um;
    p->zhi *= p->um;

    for (int r = 0; r < h; r++) {
        band_unwrap(grid + (size_t)r * w, w, 1);
    }
    for (int c = 0; c < w; c++) {
        band_unwrap(grid + c, h, w);
    }
    known = 0;
    for (size_t i = 0; i < cells; i++) {
        grid[i] /= 2 * M_PI;
        if (mesh[i] >= 0) {
            rows[known++] = nearbyint(grid[i]);
        }
    }
    double anchor = band->winding - layout_median(rows, known);
    for (size_t i = 0; i < cells; i++) {
        grid[i] = (grid[i] + anchor) * curl;
    }
    band->place = grid[(size_t)(int)band->anchor[1] * w + (int)band->anchor[0]];
    size_t measured = 0;
    for (int r = 2; r + 2 < h; r++) {
        for (int c = 2; c + 2 < w; c++) {
            int good = 1;
            for (int y = -2; good && y <= 2; y++) {
                int span = 2 - abs(y);
                for (int x = -span; good && x <= span; x++) {
                    good = mesh[(size_t)(r + y) * w + c + x] >= 0;
                }
            }
            if (!good) {
                continue;
            }
            size_t i = (size_t)r * w + c;
            double dr = (grid[i + w] - grid[i - w]) * .5;
            double dc = (grid[i + 1] - grid[i - 1]) * .5;
            double along = fabs(dc * right_run + dr * right_rise) / p->side;
            if (along > 1e-5) {
                speeds[measured++] = (BandSpeed){grid[i], 1. / along};
            }
        }
    }
    free(data);
    return measured;
}

static int band_arc(BandPiece *band, size_t pieces, const BandSpeed *speed, size_t count) {
    if (!count) {
        return 0;
    }
    double min = INFINITY, max = -INFINITY;
    for (size_t i = 0; i < count; i++) {
        min = fmin(min, speed[i].turn);
        max = fmax(max, speed[i].turn);
    }
    double first = floor(min), stop = ceil(max) + .02;
    size_t bins = (size_t)ceil((stop - first) / .02);
    double *data = layout_array(bins * 5, sizeof(*data));
    double *edges = data, *known = edges + bins, *medians = known + bins;
    double *length = medians + bins, *arc = length + bins;
    double step = (first + .02) - first;
    for (size_t b = 0; b < bins; b++) {
        edges[b] = first + b * step;
    }
    LayoutValue *values = layout_array(count, sizeof(*values));
    for (size_t i = 0; i < count; i++) {
        size_t lo = 0, hi = bins;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (edges[mid] <= speed[i].turn) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        values[i] = (LayoutValue){lo - 1, i, speed[i].length, 0};
    }
    qsort(values, count, sizeof(*values), layout_value_order);
    size_t represented = 0;
    for (size_t i = 0, j; i < count; i = j) {
        for (j = i + 1; j < count && values[j].rank == values[i].rank; j++) {
        }
        known[represented] = values[i].rank;
        medians[represented++] =
            (values[i + (j - i - 1) / 2].value + values[i + (j - i) / 2].value) / 2;
    }
    for (size_t b = 0; b < bins; b++) {
        length[b] = interpolate_layout(known, medians, represented, b);
    }
    for (size_t b = 1; b < bins; b++) {
        double around[5];
        for (int i = -2; i <= 2; i++) {
            long at = (long)b - 1 + i;
            size_t j = at < 0 ? 0 : (size_t)at >= bins ? bins - 1 : (size_t)at;
            around[i + 2] = length[j];
        }
        arc[b] = arc[b - 1] + layout_median(around, 5) * .02;
    }
    for (size_t p = 0; p < pieces; p++) {
        band[p].place = interpolate_layout(edges, arc, bins, band[p].place);
    }
    free(values);
    free(data);
    return 1;
}

static int write_band_layout(
    const char *sample, const char *scan, const char *unit, const char *output, const char *sheet) {
    json_object *record = read_record(unit, "record.json");
    json_object *rows = json_object_object_get(record, "pieces");
    int curl = json_object_get_int(json_object_object_get(record, "chirality")), status = 1;
    size_t count = 0, capacity = 0;
    BandSpeed *speed = NULL;
    if (!json_object_is_type(rows, json_type_array) || abs(curl) != 1) {
        json_object_put(record);
        return 1;
    }
    size_t length = json_object_array_length(rows);
    PlacedPiece *pieces = layout_array(length, sizeof(*pieces));
    HxLayoutPiece *places = layout_array(length, sizeof(*places));
    BandPiece *band = layout_array(length, sizeof(*band));
    for (size_t i = 0; i < length; i++) {
        json_object *row = json_object_array_get_idx(rows, i);
        if (!json_object_get_boolean(json_object_object_get(row, "written"))) {
            continue;
        }
        PlacedPiece *p = pieces + count;
        BandPiece *b = band + count++;
        p->id = json_object_get_int64(json_object_object_get(row, "piece"));
        p->area = json_object_get_double(json_object_object_get(row, "area_mm2"));
        b->winding = json_object_get_double(json_object_object_get(row, "winding"));
        char *folder = NULL;
        if (asprintf(&folder, "%s/piece-%04lld", unit, (long long)p->id) < 0) {
            goto done;
        }
        uint32_t w, h;
        int dtype;
        b->mesh = read_raw_mesh(folder, &w, &h, &dtype);
        json_object *meta = read_record(folder, "meta.json");
        free(folder);
        double voxel = json_object_get_double(json_object_object_get(meta, "voxel_um"));
        p->stride = json_object_get_int(json_object_object_get(meta, "stride"));
        json_object_put(meta);
        if (!b->mesh || w < 2 || h < 2 || w > INT_MAX - 4 || h > INT_MAX - 4 || voxel <= 0 ||
            p->stride <= 0) {
            goto done;
        }
        p->um = voxel / 1000.;
        p->side = p->stride * voxel / 1000.;
        p->width = w;
        p->height = h;
        capacity += (size_t)w * h;
    }
    double centre[2];
    if (!count || !band_centre(pieces, band, count, centre)) {
        goto done;
    }
    speed = layout_array(capacity, sizeof(*speed));
    double top = -INFINITY, low = INFINITY;
    size_t measured = 0;
    for (size_t i = 0; i < count; i++) {
        PlacedPiece *p = pieces + i;
        measured += band_pose(p, band + i, centre, curl, speed + measured);
        top = fmax(top, p->zhi);
        low = fmin(low, p->zlo);
    }
    if (!band_arc(band, count, speed, measured)) {
        goto done;
    }
    double end = 0;
    for (size_t i = 0; i < count; i++) {
        end = fmax(end, band[i].place);
    }
    end += 40;
    for (size_t i = 0; i < count; i++) {
        PlacedPiece *p = pieces + i;
        int step = fmax(1, nearbyint(1. / p->side));
        int h = (p->height - 1) / step + 1, w = (p->width - 1) / step + 1;
        p->mask = layout_array((size_t)h * w, 1);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                p->mask[(size_t)y * w + x] =
                    band[i].mesh[(size_t)(y * step) * p->width + x * step] >= 0;
            }
        }
        places[i] = (HxLayoutPiece){.mask = p->mask,
            .row_stride = w,
            .column_stride = 1,
            .rows = h,
            .columns = w,
            .group = i,
            .area = p->area,
            .group_area = p->area,
            .x = end - band[i].place - joint_x(p, band[i].anchor[0]),
            .y = curl > 0 ? top - p->zhi : p->zlo - low};
        orient_mask(p, places + i);
    }
    size_t kept = 0;
    int moved = hx_pack_layout(places, count, 60, &kept);
    if (moved < 0) {
        goto done;
    }
    json_object *items = json_object_new_array_ext(count);
    const char *name = strrchr(unit, '/') + 1;
    double width = 0;
    for (size_t i = 0; i < count; i++) {
        layout_record_piece(items, pieces + i, places + i, name, 0, 1);
        width = fmax(width, places[i].x);
    }
    json_object *summary = json_object_new_object();
    json_object_object_add(summary, "sample", json_object_new_string(sample));
    json_object_object_add(summary, "volume", json_object_new_string(scan));
    json_object_object_add(summary, "chirality", json_object_new_int(curl));
    json_object_object_add(summary, "joints", json_object_new_int(0));
    json_object_object_add(summary, "pieces", json_object_new_int64(count));
    json_object_object_add(summary, "moved", json_object_new_int(moved));
    json_object_object_add(summary, "width_mm", json_object_new_double(width));
    json_object *document = json_object_new_object();
    json_object_object_add(document, "record", summary);
    json_object_object_add(document, "pieces", items);
    if (make_folders(sheet) && !record_write_layout(output, document)) {
        puts(json_object_to_json_string_ext(summary, JSON_C_TO_STRING_PLAIN));
        status = 0;
    }
    json_object_put(document);
done:
    for (size_t i = 0; i < count; i++) {
        free(pieces[i].mask);
        free(band[i].mesh);
    }
    free(speed);
    free(band);
    free(places);
    free(pieces);
    json_object_put(record);
    return status;
}
