/* This is a GPU program, written in a language called GLSL (version 1.1), that is
 * responsible for taking the raw bayer data from the image sensor and rendering it
 * in a way that is visually understandable. For some types of optical experiments,
 * it may be desireable to modify the way this shader works. For most other use cases,
 * the most you may want to do with this program is to modify the values of the variables.
 * For that, you can use the user interface.
 */

/* 0 = convert to RGB
 * 1 = show raw bayer intensity values
 */
#define SHOW_RAW_BAYER 0

varying highp vec2 texcoords;

uniform highp sampler2D bayer_data;

/* Size of the texture (ex. 3280x2464) */
uniform highp vec2 texture_size;

/* Size of one texel (ex. 1.0/3280.0, 1.0/2464.0) */
uniform highp vec2 texel_size;

/* The weight of each color value. */
uniform highp vec3 color_balance;

/* The digital gain to apply to all color channels. */
uniform highp float gain;

/* Gamma correction exponent. */
uniform highp float gamma;

highp vec3
filter_rgb(highp vec3 rgb)
{
  rgb = rgb * color_balance * gain;
  rgb = pow(rgb, vec3(1.0 / gamma));
  return rgb;
}

/* * *
 * Everything below this point is related to unpacking the bayer data.
 * * */

/* Converts two 8-bit components into a normalized 10-bit float */

highp float
unpack_10bit(highp float low, highp float high)
{
    return (floor(low * 255.0) + floor(high * 255.0) * 256.0) / 1023.0;
}

/**
 * @brief This converts normal texture coordinates [0.0, 1.0) to texture coordinates
 *        in the bayer texture, which is also [0.0, 1.0). What this function does is
 *        avoid the circumstance where we need to fetch the 2nd pixel packed in the
 *        RGBA format, but rounding by GL_NEAREST causes us to go to the next column.
 *        This function ensures the coordinate will always be rounded down to the
 *        proper texel.
 * */
highp vec2
to_bayer_xy(highp vec2 xy)
{
  highp float sx = texture_size.x * 0.5;
  highp float x1 = floor(xy.x * sx) / sx;
  return vec2(x1, xy.y);
}

highp float
fetch_bayer_pixel(highp float x, highp float y)
{
    highp vec4 texel = texture2D(bayer_data, to_bayer_xy(vec2(x, y)));

    highp float low;
    highp float high;

    if (mod(x * texture_size.x, 2.0) < 1.0) {
        low = texel.r;
        high = texel.g;
    } else {
        low = texel.b;
        high = texel.a;
    }

    return unpack_10bit(low, high);
}

highp vec3
fetch_rgb(highp vec2 uv)
{
    highp vec2 pos = uv * texture_size;

    highp float tx = floor(pos.x);
    highp float ty = floor(pos.y);

    highp float idx = mod(ty, 2.0) * 2.0 + mod(tx, 2.0);

    highp float r = 0.0;
    highp float g = 0.0;
    highp float b = 0.0;

    highp float x = uv.x;
    highp float y = uv.y;
    highp float dx = texel_size.x;
    highp float dy = texel_size.y;

    /* Note: This interpolation assumes that the borders of the image
     *       are mirrored with GL_MIRRORED_REPEAT. Without this behavior,
     *       the interpolation would incorrectly blend the color channels
     *       at the border of the image.
     *
     *       This pattern assumes RGGB layout.
     * */

    if (idx < 0.5) {
        /* Green */
        g = fetch_bayer_pixel(x, y);
        b = 0.5 * (fetch_bayer_pixel(x - dx, y) + fetch_bayer_pixel(x + dx, y));
        r = 0.5 * (fetch_bayer_pixel(x, y - dy) + fetch_bayer_pixel(x, y + dy));
    } else if (idx < 1.5) {
        /* Blue */
        b = fetch_bayer_pixel(x, y);
        g = 0.25 * (fetch_bayer_pixel(x - dx, y) + fetch_bayer_pixel(x + dx, y) + fetch_bayer_pixel(x, y - dy) + fetch_bayer_pixel(x, y + dy));
        r = 0.25 * (fetch_bayer_pixel(x - dx, y - dy) + fetch_bayer_pixel(x + dx, y - dy) + fetch_bayer_pixel(x - dx, y + dy) + fetch_bayer_pixel(x + dx, y + dy));
    } else if (idx < 2.5) {
        /* Red */
        r = fetch_bayer_pixel(x, y);
        g = 0.25 * (fetch_bayer_pixel(x - dx, y) + fetch_bayer_pixel(x + dx, y) + fetch_bayer_pixel(x, y - dy) + fetch_bayer_pixel(x, y + dy));
        b = 0.25 * (fetch_bayer_pixel(x - dx, y - dy) + fetch_bayer_pixel(x + dx, y - dy) + fetch_bayer_pixel(x - dx, y + dy) + fetch_bayer_pixel(x + dx, y + dy));
    } else if (idx < 3.5) {
        /* Green */
        g = fetch_bayer_pixel(x, y);
        r = 0.5 * (fetch_bayer_pixel(x - dx, y) + fetch_bayer_pixel(x + dx, y));
        b = 0.5 * (fetch_bayer_pixel(x, y - dy) + fetch_bayer_pixel(x, y + dy));
    }

    return vec3(r, g, b);
}

void
main()
{
#if SHOW_RAW_BAYER
  highp float c = fetch_bayer_pixel(texcoords.x, texcoords.y);
  gl_FragColor = vec4(c, c, c, 1.0);
#else
  gl_FragColor = vec4(filter_rgb(fetch_rgb(texcoords)), 1.0);
#endif
}
