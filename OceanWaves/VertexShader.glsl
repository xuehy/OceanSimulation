#version 410 core
layout(location = 0) in vec3 vertex;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 texture;

uniform mat4 Projection;
uniform mat4 View;
uniform mat4 Model;
uniform vec3 light_direction;
 
out vec3 light_dir;
out vec3 normal_vector;
out vec3 FragPos;
//out float fog_factor;
out vec2 tex_coord;
 
void main() {
    gl_Position = Projection * View * Model * vec4(vertex, 1.0);
    //fog_factor = min(-gl_Position.z/500.0, 1.0);
 
    light_dir = normalize(light_direction);
    normal_vector = vec3(inverse(transpose(Model)) * vec4(normalize(normal), 0.0));
    FragPos = vec3(Model * vec4(vertex, 1.0));
    tex_coord = texture.xy;
}