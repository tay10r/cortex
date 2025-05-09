attribute highp vec2 a_position;

varying highp vec2 texcoords;

void
main()
{
  texcoords = a_position;

  gl_Position = vec4(a_position * 2.0 - 1.0, 0.0, 1.0);
}
