#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "packing.h"

enum { SHIFT_STEP = 5, MOST_SHIFT = 1500, ALIGNMENT = 64 };

typedef struct {
    uint64_t *bits;
    size_t words, cells;
    int rows, columns;
} Mask;

typedef struct {
    size_t index, order;
    int group;
    double area;
} Member;

typedef struct {
    size_t first, count, order;
    double area;
} Group;

int hx_layout_outward(const double *xyz, size_t rows, size_t columns, double x, double y) {
    if (rows < 3 || columns < 3)
        return 0;
    size_t plane = rows * columns;
    const double *yy = xyz + plane, *zz = yy + plane;
    double vote = 0;
    for (size_t row = 1; row + 1 < rows; row++) {
        size_t end = (row + 1) * columns - 1;
        for (size_t i = row * columns + 1; i < end; i++) {
            if (!(xyz[i] >= 0 && xyz[i - 1] >= 0 && xyz[i + 1] >= 0 && xyz[i - columns] >= 0 &&
                    xyz[i + columns] >= 0))
                continue;
            double cx = (xyz[i + 1] - xyz[i - 1]) * 0.5;
            double cy = (yy[i + 1] - yy[i - 1]) * 0.5;
            double cz = (zz[i + 1] - zz[i - 1]) * 0.5;
            double rx = (xyz[i + columns] - xyz[i - columns]) * 0.5;
            double ry = (yy[i + columns] - yy[i - columns]) * 0.5;
            double rz = (zz[i + columns] - zz[i - columns]) * 0.5;
            double nx = cy * rz - cz * ry, ny = cz * rx - cx * rz;
            double dx = x - xyz[i], dy = y - yy[i];
            double scale = sqrt(nx * nx + ny * ny) * sqrt(dx * dx + dy * dy);
            vote += (nx * dx + ny * dy) / (scale < 1e-9 ? 1e-9 : scale);
        }
    }
    return vote < 0;
}

void hx_layout_flips(int rows_up, int outward, int curl, int *flip_rows, int *flip_cols) {
    *flip_rows = rows_up == (curl > 0);
    *flip_cols = *flip_rows == outward;
}

static size_t word_count(int columns) { return ((size_t)columns + ALIGNMENT - 1) / ALIGNMENT; }

static uint64_t *pack_masks(const HxLayoutPiece *pieces, size_t count, Mask *masks) {
    size_t total = 0;
    for (size_t i = 0; i < count; i++) {
        const HxLayoutPiece *p = pieces + i;
        size_t words = word_count(p->columns);
        if (words && (size_t)p->rows > (SIZE_MAX / sizeof(uint64_t) - total) / words) {
            return NULL;
        }
        total += words * p->rows;
    }
    uint64_t *storage = calloc(total ? total : 1, sizeof(*storage));
    if (!storage) {
        return NULL;
    }
    uint64_t *bits = storage;
    for (size_t i = 0; i < count; i++) {
        const HxLayoutPiece *p = pieces + i;
        Mask mask = {
            .bits = bits, .words = word_count(p->columns), .rows = p->rows, .columns = p->columns};
        for (int y = 0; y < p->rows; y++) {
            const uint8_t *row = p->mask + (ptrdiff_t)y * p->row_stride;
            int x = 0;
            for (size_t w = 0; w < mask.words; w++) {
                uint64_t word = 0;
                int remaining = p->columns - x;
                int length = remaining < ALIGNMENT ? remaining : ALIGNMENT;
                for (int k = 0; k < length; k++, x++) {
                    word |= (uint64_t)(row[(ptrdiff_t)x * p->column_stride] != 0) << k;
                }
                bits[w] = word;
                mask.cells += __builtin_popcountll(word);
            }
            bits += mask.words;
        }
        masks[i] = mask;
    }
    return storage;
}

static size_t overlap(const Mask *sheet, const Mask *mask, int x, int y, size_t limit) {
    size_t count = 0;
    unsigned shift = x % ALIGNMENT;
    const uint64_t *old = sheet->bits + (size_t)y * sheet->words + x / ALIGNMENT;
    const uint64_t *bits = mask->bits;
    for (int r = 0; r < mask->rows && count <= limit; r++) {
        for (size_t w = 0; w < mask->words; w++) {
            uint64_t occupied = old[w] >> shift;
            if (shift) {
                occupied |= old[w + 1] << (ALIGNMENT - shift);
            }
            count += __builtin_popcountll(occupied & bits[w]);
        }
        old += sheet->words;
        bits += mask->words;
    }
    return count;
}

static void occupy(Mask *sheet, const Mask *mask, int x, int y) {
    unsigned shift = x % ALIGNMENT;
    uint64_t *old = sheet->bits + (size_t)y * sheet->words + x / ALIGNMENT;
    const uint64_t *bits = mask->bits;
    for (int r = 0; r < mask->rows; r++) {
        for (size_t w = 0; w < mask->words; w++) {
            old[w] |= bits[w] << shift;
            if (shift) {
                old[w + 1] |= bits[w] >> (ALIGNMENT - shift);
            }
        }
        old += sheet->words;
        bits += mask->words;
    }
}

static int by_group(const void *left, const void *right) {
    const Member *a = left, *b = right;
    if (a->group != b->group) {
        return a->group < b->group ? -1 : 1;
    }
    return (a->order > b->order) - (a->order < b->order);
}

static int heavier(double a, double b, size_t i, size_t j) {
    if (a != b) {
        return a > b ? -1 : 1;
    }
    return (i > j) - (i < j);
}

static int by_weight(const void *left, const void *right) {
    const Member *a = left, *b = right;
    return heavier(a->area, b->area, a->order, b->order);
}

static int by_group_weight(const void *left, const void *right) {
    const Group *a = left, *b = right;
    return heavier(a->area, b->area, a->order, b->order);
}

static int row_of(const Mask *sheet, const Mask *mask, double y) {
    int limit = sheet->rows - mask->rows - 1;
    return y <= 0 ? 0 : y >= limit ? limit : (int)y;
}

static int column_of(double x) { return (int)x + MOST_SHIFT + 10; }

static int fits(const Mask *sheet, const Mask *mask, int x) {
    return x >= 0 && x < sheet->columns - mask->columns;
}

static int free_enough(const Mask *sheet, const Mask *mask, int x, int y) {
    size_t limit = mask->cells / 10;
    return fits(sheet, mask, x) && (!mask->cells || overlap(sheet, mask, x, y, limit) <= limit);
}

static int next_shift(int shift, int step) { return shift > 0 ? -shift : step - shift; }

static int best_shift(const Mask *sheet, const Mask *masks, const HxLayoutPiece *pieces,
    const Member *members, size_t count) {
    int best = 0;
    double least = INFINITY;
    for (int shift = 0; shift < MOST_SHIFT && least != 0;
        shift = next_shift(shift, 8 * SHIFT_STEP)) {
        double hit = 0;
        for (size_t j = 0; j < count && hit < least - 1e-6; j++) {
            size_t i = members[j].index;
            const Mask *mask = masks + i;
            int x = column_of(pieces[i].x) + shift;
            int y = row_of(sheet, mask, pieces[i].y);
            if (mask->cells && fits(sheet, mask, x)) {
                hit += (double)overlap(sheet, mask, x, y, SIZE_MAX) / mask->cells * pieces[i].area;
            }
        }
        if (hit < least - 1e-6) {
            best = shift;
            least = hit;
        }
    }
    return best;
}

static int move_one(Mask *sheet, const Mask *mask, HxLayoutPiece *piece) {
    int y = row_of(sheet, mask, piece->y), origin = column_of(piece->x);
    int limit = 4 * sheet->columns * SHIFT_STEP;
    for (int shift = 0; shift < limit; shift = next_shift(shift, SHIFT_STEP)) {
        int x = origin + shift;
        if (free_enough(sheet, mask, x, y)) {
            occupy(sheet, mask, x, y);
            piece->shift = shift;
            piece->shifted = 1;
            piece->x += shift;
            return abs(shift) > 2 * SHIFT_STEP;
        }
    }
    return 0;
}

static int place_all(HxLayoutPiece *pieces, size_t count, Mask *sheet, const Mask *masks,
    Member *members, Group *groups, Member *loose, size_t *kept) {
    for (size_t i = 0; i < count; i++) {
        members[i] = (Member){i, i, pieces[i].group, pieces[i].area};
    }
    qsort(members, count, sizeof(*members), by_group);
    size_t group_count = 0;
    for (size_t start = 0; start < count;) {
        size_t end = start + 1;
        while (end < count && members[end].group == members[start].group) {
            end++;
        }
        size_t i = members[start].index;
        groups[group_count++] = (Group){start, end - start, i, pieces[i].group_area};
        start = end;
    }
    qsort(groups, group_count, sizeof(*groups), by_group_weight);
    size_t loose_count = 0;
    for (size_t j = 0; j < group_count; j++) {
        Group group = groups[j];
        Member *joined = members + group.first;
        int shift = 0;
        if (group.count > 1) {
            shift = best_shift(sheet, masks, pieces, joined, group.count);
            qsort(joined, group.count, sizeof(*joined), by_weight);
        }
        for (size_t n = 0; n < group.count; n++) {
            size_t i = joined[n].index;
            HxLayoutPiece *p = pieces + i;
            const Mask *mask = masks + i;
            p->x += shift;
            int x = column_of(p->x), y = row_of(sheet, mask, p->y);
            if (group.count > 1 && free_enough(sheet, mask, x, y)) {
                occupy(sheet, mask, x, y);
            } else {
                loose[loose_count] = (Member){i, loose_count, 0, p->area};
                loose_count++;
            }
        }
    }
    qsort(loose, loose_count, sizeof(*loose), by_weight);
    int moved = 0;
    for (size_t n = 0; n < loose_count; n++) {
        size_t i = loose[n].index;
        moved += move_one(sheet, masks + i, pieces + i);
    }
    *kept = count - loose_count;
    return moved;
}

int hx_pack_layout(HxLayoutPiece *pieces, size_t count, double padding, size_t *kept) {
    if (!pieces || !kept || !count || count > INT_MAX || !isfinite(padding)) {
        return -1;
    }
    double left = INFINITY, right = -INFINITY, bottom = -INFINITY;
    int tallest = 0;
    for (size_t i = 0; i < count; i++) {
        HxLayoutPiece *p = pieces + i;
        if (p->rows < 0 || p->columns < 0 || (!p->mask && p->rows) || !isfinite(p->x) ||
            !isfinite(p->y) || !(p->area >= 0 && isfinite(p->area)) || !isfinite(p->group_area)) {
            return -1;
        }
        left = fmin(left, p->x);
        right = fmax(right, p->x);
        bottom = fmax(bottom, p->y);
        tallest = p->rows > tallest ? p->rows : tallest;
    }
    double width = right - left, height = bottom + padding;
    if (!(width < INT_MAX / 32 - 9020 && height >= 0 && height < INT_MAX - (double)tallest - 4)) {
        return -1;
    }
    Mask sheet = {
        .rows = (int)height + 4 + tallest, .columns = (int)width + 6000 + 2 * (MOST_SHIFT + 10)};
    sheet.words = word_count(sheet.columns) + 1;
    if (sheet.words > SIZE_MAX / sizeof(*sheet.bits) / (size_t)sheet.rows) {
        return -1;
    }
    sheet.bits = calloc((size_t)sheet.rows * sheet.words, sizeof(*sheet.bits));
    Mask *masks = calloc(count, sizeof(*masks));
    Member *members = calloc(count, sizeof(*members)), *loose = calloc(count, sizeof(*loose));
    Group *groups = calloc(count, sizeof(*groups));
    uint64_t *bits = masks ? pack_masks(pieces, count, masks) : NULL;
    int moved = -1;
    if (sheet.bits && bits && masks && members && loose && groups) {
        for (size_t i = 0; i < count; i++) {
            pieces[i].x -= left;
            pieces[i].from_joint = pieces[i].x;
            pieces[i].shifted = 0;
        }
        moved = place_all(pieces, count, &sheet, masks, members, groups, loose, kept);
        left = INFINITY;
        for (size_t i = 0; i < count; i++) {
            left = fmin(left, pieces[i].x);
        }
        for (size_t i = 0; i < count; i++) {
            pieces[i].x -= left;
            pieces[i].from_joint = pieces[i].x - pieces[i].from_joint;
        }
    }
    free(bits);
    free(groups);
    free(loose);
    free(members);
    free(masks);
    free(sheet.bits);
    return moved;
}

typedef struct {
    size_t target;
    int weight, sign;
    double x, y;
} Link;

typedef struct {
    size_t target, rank;
    int weight, sign;
    double x, y;
} Candidate;

typedef struct {
    size_t *first;
    Link *links;
    Candidate *heap;
    const size_t *rank;
    size_t queued;
} LayoutGraph;

static void free_graph(LayoutGraph *graph) {
    free(graph->heap);
    free(graph->links);
    free(graph->first);
}

static int make_graph(LayoutGraph *graph, size_t count, const size_t *rank,
    const HxLayoutJoin *joins, size_t join_count) {
    if (join_count > SIZE_MAX / (2 * sizeof(Candidate))) {
        return 0;
    }
    graph->rank = rank;
    graph->first = calloc(count + 1, sizeof(*graph->first));
    graph->links = calloc(join_count ? 2 * join_count : 1, sizeof(*graph->links));
    graph->heap = calloc(join_count ? 2 * join_count : 1, sizeof(*graph->heap));
    if (!graph->first || !graph->links || !graph->heap) {
        return 0;
    }
    for (size_t i = 0; i < join_count; i++) {
        HxLayoutJoin j = joins[i];
        if (j.a >= count || j.b >= count || !isfinite(j.x) || !isfinite(j.y)) {
            return 0;
        }
        graph->first[j.a + 1]++;
        graph->first[j.b + 1]++;
    }
    for (size_t i = 0; i < count; i++) {
        graph->first[i + 1] += graph->first[i];
    }
    for (size_t i = 0; i < join_count; i++) {
        HxLayoutJoin j = joins[i];
        graph->links[graph->first[j.a]++] = (Link){j.b, j.weight, j.sign, j.x, j.y};
        graph->links[graph->first[j.b]++] = (Link){j.a, j.weight, j.sign, -j.x, -j.y};
    }
    memmove(graph->first + 1, graph->first, count * sizeof(*graph->first));
    graph->first[0] = 0;
    return 1;
}

static int comes_first(Candidate a, Candidate b) {
    if (a.weight != b.weight) {
        return a.weight > b.weight;
    }
    if (a.rank != b.rank) {
        return a.rank < b.rank;
    }
    if (a.x != b.x) {
        return a.x < b.x;
    }
    if (a.y != b.y) {
        return a.y < b.y;
    }
    return a.sign < b.sign;
}

static void offer(LayoutGraph *graph, Candidate value) {
    size_t index = graph->queued++;
    while (index) {
        size_t parent = (index - 1) / 2;
        if (!comes_first(value, graph->heap[parent])) {
            break;
        }
        graph->heap[index] = graph->heap[parent];
        index = parent;
    }
    graph->heap[index] = value;
}

static Candidate take(LayoutGraph *graph) {
    Candidate result = graph->heap[0], tail = graph->heap[--graph->queued];
    size_t index = 0;
    for (size_t child = 1; child < graph->queued; child = 2 * index + 1) {
        if (child + 1 < graph->queued && comes_first(graph->heap[child + 1], graph->heap[child])) {
            child++;
        }
        if (!comes_first(graph->heap[child], tail)) {
            break;
        }
        graph->heap[index] = graph->heap[child];
        index = child;
    }
    graph->heap[index] = tail;
    return result;
}

static void neighbours(LayoutGraph *graph, const uint8_t *seen, Candidate at) {
    for (size_t i = graph->first[at.target]; i < graph->first[at.target + 1]; i++) {
        Link link = graph->links[i];
        if (!seen[link.target]) {
            offer(graph,
                (Candidate){link.target, graph->rank[link.target], link.weight, at.sign * link.sign,
                    at.x + link.x, at.y + link.y});
        }
    }
}

static void store_member(
    HxLayoutPiece *p, uint8_t *seen, size_t *order, size_t *used, Candidate step, int group) {
    seen[step.target] = 1;
    order[(*used)++] = step.target;
    p->group = group;
    p->x = step.x;
    p->y = step.y;
}

typedef struct {
    int top, bottom;
    size_t first, end;
} Occupied;

static void mark(Mask *sheet, const Mask *mask, int x, int y, Occupied *used) {
    occupy(sheet, mask, x, y);
    size_t first = x / ALIGNMENT, end = first + mask->words + (x % ALIGNMENT != 0);
    used->first = first < used->first ? first : used->first;
    used->end = end > used->end ? end : used->end;
    used->top = y < used->top ? y : used->top;
    used->bottom = y + mask->rows > used->bottom ? y + mask->rows : used->bottom;
}

static int grow_mosaics(HxLayoutPiece *pieces, size_t count, LayoutGraph *graph, const Mask *masks,
    const Member *seeds, uint8_t *seen, Mask *sheet, size_t *order) {
    const int half_w = sheet->columns / 2, half_h = sheet->rows / 2;
    size_t used = 0;
    int groups = 0;
    for (size_t n = 0; n < count; n++) {
        size_t i = seeds[n].index;
        if (seen[i]) {
            continue;
        }
        if (!fits(sheet, masks + i, half_w) || masks[i].rows >= half_h) {
            return -1;
        }
        Candidate anchor = {.target = i, .sign = 1};
        Occupied painted = {.top = sheet->rows, .first = sheet->words};
        store_member(pieces + i, seen, order, &used, anchor, groups);
        neighbours(graph, seen, anchor);
        if (graph->queued) {
            mark(sheet, masks + i, half_w, half_h, &painted);
        }
        while (graph->queued) {
            Candidate step = take(graph);
            if (seen[step.target] ||
                !(step.x > -half_w - 1 && step.x < half_w && step.y > -half_h - 1 &&
                    step.y < half_h)) {
                continue;
            }
            const Mask *mask = masks + step.target;
            int x = (int)step.x + half_w, y = (int)step.y + half_h;
            if (y < sheet->rows - mask->rows && free_enough(sheet, mask, x, y)) {
                mark(sheet, mask, x, y, &painted);
                store_member(pieces + step.target, seen, order, &used, step, groups);
                neighbours(graph, seen, step);
            }
        }
        for (int y = painted.top; y < painted.bottom; y++) {
            uint64_t *row = sheet->bits + (size_t)y * sheet->words + painted.first;
            memset(row, 0, (painted.end - painted.first) * sizeof(*row));
        }
        groups++;
    }
    return groups;
}

int hx_layout_mosaics(HxLayoutPiece *pieces, size_t count, const size_t *rank,
    const HxLayoutJoin *joins, size_t join_count, size_t *order) {
    if (!count) {
        return 0;
    }
    if (count > INT_MAX) {
        return -1;
    }
    Mask sheet = {.rows = 500, .columns = 4000};
    sheet.words = word_count(sheet.columns) + 1;
    for (size_t i = 0; i < count; i++) {
        if (pieces[i].rows < 0 || pieces[i].columns < 0 || (!pieces[i].mask && pieces[i].rows) ||
            !isfinite(pieces[i].area)) {
            return -1;
        }
    }
    LayoutGraph graph = {0};
    Mask *masks = calloc(count, sizeof(*masks));
    uint64_t *bits = masks ? pack_masks(pieces, count, masks) : NULL;
    uint8_t *seen = calloc(count, sizeof(*seen));
    Member *seeds = calloc(count, sizeof(*seeds));
    sheet.bits = calloc((size_t)sheet.rows * sheet.words, sizeof(*sheet.bits));
    int groups = -1;
    if (masks && bits && seen && seeds && sheet.bits &&
        make_graph(&graph, count, rank, joins, join_count)) {
        for (size_t i = 0; i < count; i++) {
            seeds[i] = (Member){.index = i, .order = i, .area = pieces[i].area};
        }
        qsort(seeds, count, sizeof(*seeds), by_weight);
        groups = grow_mosaics(pieces, count, &graph, masks, seeds, seen, &sheet, order);
    }
    free_graph(&graph);
    free(sheet.bits);
    free(seeds);
    free(seen);
    free(bits);
    free(masks);
    return groups;
}

static void follow_turns(LayoutGraph *graph, const double *areas, const Member *seeds, size_t count,
    uint8_t *seen, size_t *group, int *turns) {
    for (size_t n = 0; n < count; n++) {
        size_t i = seeds[n].index;
        if (seen[i]) {
            continue;
        }
        size_t used = 1;
        seen[i] = 1;
        turns[i] = 1;
        group[0] = i;
        double area = areas[i];
        neighbours(graph, seen, (Candidate){.target = i, .sign = 1});
        while (graph->queued) {
            Candidate step = take(graph);
            if (!seen[step.target]) {
                seen[step.target] = 1;
                turns[step.target] = step.sign;
                group[used++] = step.target;
                area += areas[step.target] * step.sign;
                neighbours(graph, seen, step);
            }
        }
        if (area < 0) {
            for (size_t j = 0; j < used; j++) {
                turns[group[j]] = -turns[group[j]];
            }
        }
    }
}

int hx_layout_turns(const double *areas, size_t count, const size_t *rank,
    const HxLayoutJoin *joins, size_t join_count, int *turns) {
    if (!count) {
        return 1;
    }
    if (count > INT_MAX) {
        return 0;
    }
    LayoutGraph graph = {0};
    uint8_t *seen = calloc(count, sizeof(*seen));
    size_t *group = calloc(count, sizeof(*group));
    Member *seeds = calloc(count, sizeof(*seeds));
    int ok = seen && group && seeds && make_graph(&graph, count, rank, joins, join_count);
    if (ok) {
        for (size_t i = 0; i < count; i++) {
            seeds[i] = (Member){.index = i, .order = i, .area = areas[i]};
        }
        qsort(seeds, count, sizeof(*seeds), by_weight);
        follow_turns(&graph, areas, seeds, count, seen, group, turns);
    }
    free_graph(&graph);
    free(seeds);
    free(group);
    free(seen);
    return ok;
}
