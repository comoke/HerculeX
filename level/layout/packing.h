#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct {
    const uint8_t *mask;
    ptrdiff_t row_stride, column_stride;
    int rows, columns, group, shifted;
    double area, group_area, x, y, shift, from_joint;
} HxLayoutPiece;

typedef struct {
    size_t a, b;
    int weight, sign;
    double x, y;
} HxLayoutJoin;

int hx_layout_outward(const double *xyz, size_t rows, size_t columns, double x, double y);
void hx_layout_flips(int rows_up, int outward, int curl, int *flip_rows, int *flip_cols);
int hx_pack_layout(HxLayoutPiece *pieces, size_t count, double padding, size_t *kept);
int hx_layout_mosaics(HxLayoutPiece *pieces, size_t count, const size_t *rank,
    const HxLayoutJoin *joins, size_t join_count, size_t *order);
int hx_layout_turns(const double *areas, size_t count, const size_t *rank,
    const HxLayoutJoin *joins, size_t join_count, int *turns);
