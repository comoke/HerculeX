#include <stdio.h>
#include "../level/layout/packing.c"

static void require(int ok, const char *message) {
    if (!ok) {
        fprintf(stderr, "layout golden: %s\n", message);
        exit(1);
    }
}

int main(void) {
    uint8_t mask[400];
    memset(mask, 1, sizeof(mask));
    HxLayoutPiece pieces[4] = {0};
    size_t rank[] = {0, 1, 2, 3}, order[4];
    for (int i = 0; i < 4; i++) {
        pieces[i] = (HxLayoutPiece){.mask = mask,
            .rows = 20,
            .columns = 20,
            .row_stride = 20,
            .column_stride = 1,
            .area = 400. - i};
    }
    const double step = 99. * .2;
    HxLayoutJoin joints[] = {
        {0, 1, 20, 1, step, 0}, {1, 2, 20, 1, step, 0}, {2, 3, 20, 1, -step, 0}};
    int groups = hx_layout_mosaics(pieces, 4, rank, joints, 3, order);
    require(groups == 2 && pieces[1].x == step && pieces[2].x == 2 * step &&
            pieces[0].group == pieces[1].group && pieces[0].group == pieces[2].group &&
            pieces[0].group != pieces[3].group,
        "a joint cannot fold the mosaic onto itself");

    const double x[] = {-7, -3, -7}, area[] = {12, 8, 5};
    for (int i = 0; i < 3; i++) {
        pieces[i] = (HxLayoutPiece){.mask = mask,
            .rows = 2,
            .columns = 4,
            .row_stride = 4,
            .column_stride = 1,
            .group = i == 2,
            .area = area[i],
            .group_area = i == 2 ? 5 : 20,
            .x = x[i],
            .y = 3};
    }
    size_t kept;
    require(hx_pack_layout(pieces, 3, 60, &kept) >= 0 && pieces[0].x == 5 && pieces[1].x == 9 &&
            pieces[2].x == 0 && pieces[0].y == 3 && pieces[1].y == 3 && pieces[2].y == 3 &&
            pieces[0].from_joint == 5 && pieces[1].from_joint == 5 && pieces[2].from_joint == 0,
        "packing keeps offsets relative to the shifted sheet origin");

    int turns[4];
    joints[0].weight = 50;
    joints[1].weight = 30;
    joints[1].sign = -1;
    require(hx_layout_turns((double[]){100, 90, 5, 7}, 4, rank, joints, 3, turns) &&
            turns[0] == 1 && turns[1] == 1 && turns[2] == -1 && turns[3] == -1,
        "pieces connected to the same sheet take consistent faces");
    joints[0].sign = -1;
    require(hx_layout_turns((double[]){3, 9}, 2, rank, joints, 1, turns) && turns[0] == -1 &&
            turns[1] == 1,
        "the larger piece keeps its face");

    const size_t rows = 20, cols = 30, cells = rows * cols;
    double xyz[3 * 20 * 30];
    for (int flip = 0; flip < 4; flip++) {
        int rev_r = flip & 1, rev_c = (flip >> 1) & 1;
        for (size_t r = 0; r < rows; r++) {
            size_t sr = rev_r ? rows - 1 - r : r;
            double z = 500. + 16. * sr;
            for (size_t c = 0; c < cols; c++) {
                size_t sc = rev_c ? cols - 1 - c : c;
                double theta = 0.3 + 0.6 * sc / (cols - 1);
                size_t at = r * cols + c;
                xyz[at] = 5000. + 2000. * cos(theta);
                xyz[cells + at] = 5000. + 2000. * sin(theta);
                xyz[2 * cells + at] = z;
            }
        }
        int rows_up = xyz[2 * cells + cols] > xyz[2 * cells];
        int outward = hx_layout_outward(xyz, rows, cols, 5000., 5000.);
        for (int curl = -1; curl <= 1; curl += 2) {
            int fr = 0, fc = 0;
            hx_layout_flips(rows_up, outward, curl, &fr, &fc);
            size_t r0 = fr ? rows - 1 - 10 : 10, r1 = fr ? r0 - 1 : r0 + 1;
            size_t c0 = fc ? cols - 1 - 15 : 15, c1 = fc ? c0 - 1 : c0 + 1;
            size_t p0 = r0 * cols + c0, pr = r1 * cols + c0, pc = r0 * cols + c1;
            double fx = xyz[p0] - 5000., fy = xyz[cells + p0] - 5000.;
            double rx = fy * curl, ry = -fx * curl;
            double col_right = (xyz[pc] - xyz[p0]) * rx + (xyz[cells + pc] - xyz[cells + p0]) * ry;
            double row_up = (xyz[2 * cells + pr] - xyz[2 * cells + p0]) * curl;
            require(col_right > 0 && row_up < 0,
                "a piece seen from the middle is never mirrored and curl only sets its top");
        }
    }
    return 0;
}
