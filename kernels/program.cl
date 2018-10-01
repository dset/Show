int index(int width, int row, int col);
float2 complexMultiply(float2 c1, float2 c2);

kernel void ft2rows(global float2* data, global float2* out) {
    int height = get_global_size(0);
    int width = get_global_size(1);

    int x = get_global_id(1);
    int y = get_global_id(0);

    float2 sum = (float2) (0.0f, 0.0f);
    for (int col = 0; col < width; col++) {
        float2 value = data[index(width, y, col)];
        float angle = 2 * M_PI_F * x * col / width;
        float2 dir = (float2) (cos(angle), -sin(angle));
        sum += complexMultiply(value, dir);
    }

    out[index(width, y, x)] = sum;
}

kernel void ft2cols(global float2* data, global float2* out) {
    int height = get_global_size(0);
    int width = get_global_size(1);

    int x = get_global_id(1);
    int y = get_global_id(0);

    float2 sum = (float2) (0.0f, 0.0f);
    for (int row = 0; row < height; row++) {
        float2 value = data[index(width, row, x)];
        float angle = 2 * M_PI_F * y * row / height;
        float2 dir = (float2) (cos(angle), -sin(angle));
        sum += complexMultiply(value, dir);
    }

    out[index(width, y, x)] = sum;
}

int index(int width, int row, int col) {
    return width * row + col;
}

float2 complexMultiply(float2 c1, float2 c2) {
    return (float2) (c1.x * c2.x - c1.y * c2.y, c1.x * c2.y + c1.y * c2.x);
}