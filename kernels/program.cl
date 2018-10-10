int index(int width, int channels, int row, int col, int channel);
float2 complexMultiply(float2 c1, float2 c2);

kernel void byteImageToComplex(global uchar* data, global float2* out) {
    int width = get_global_size(1);
    int channels = get_global_size(2);

    int row = get_global_id(0);
    int col = get_global_id(1);
    int channel = get_global_id(2);

    int i = index(width, channels, row, col, channel);
    out[i] = (float2) (data[i], 0);
}

kernel void fourierRowsAndTranspose(global float2* data, global float2* out) {
    int height = get_global_size(0);
    int width = get_global_size(1);
    int channels = get_global_size(2);

    int row = get_global_id(0);
    int col = get_global_id(1);
    int channel = get_global_id(2);

    float2 sum = (float2) (0.0f, 0.0f);
    for (int x = 0; x < width; x++) {
        float2 value = data[index(width, channels, row, x, channel)];
        float angle = 2 * M_PI_F * col * x / width;
        float2 dir = (float2) (cos(angle), -sin(angle));
        sum += complexMultiply(value, dir);
    }

    out[index(height, channels, col, row, channel)] = sum;
}

kernel void  complexImageToLogMagnitude(global float2* data, global float* out) {
    int width = get_global_size(1);
    int channels = get_global_size(2);

    int row = get_global_id(0);
    int col = get_global_id(1);
    int channel = get_global_id(2);

    int i = index(width, channels, row, col, channel);
    out[i] = log1p(length(data[i]));
}

int index(int width, int channels, int row, int col, int channel) {
    return row * width * channels + col * channels + channel;
}

float2 complexMultiply(float2 c1, float2 c2) {
    return (float2) (c1.x * c2.x - c1.y * c2.y, c1.x * c2.y + c1.y * c2.x);
}