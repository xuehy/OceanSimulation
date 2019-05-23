#version 410 core
in vec3 normal_vector;
in vec3 light_dir;
in vec3 FragPos;
in vec2 tex_coord;
//in float fog_factor;
uniform sampler2D water;
uniform vec3 viewPos;
uniform samplerCube sky;
out vec4 fragColor;

// ambient light + sunlight (reflection and specular) * sky (reflection and specular)
// 1. compute sunlight specular reflection and refraction weights using fresnel
// 2. compute sunlight diffuse weights by (1 - specular reflection weights)
// 3. compute sky specular reflection and refraction weights using fresnel
// 4. compute sky diffuse weights by (1 - specular reflection weights)
void main (void) {
    //fragColor = vec4(1.0, 1.0, 1.0, 1.0);
 
    vec3 norm = normalize(normal_vector);
    vec3 lightdir = normalize(light_dir);
	vec3 viewDir = normalize(viewPos - FragPos);
	// compute sunlight
    vec4 sunlight = vec4(1,1,1,1);//texture(water, tex_coord);
 
    float coeff = max(dot(norm, -lightdir), 0);
	sunlight = coeff * sunlight;

	vec4 ambient_color  = vec4(0.0, 0.65, 0.75, 1.0);
    vec4 diffuse_color  = vec4(0.5, 0.65, 0.75, 1.0);
    vec4 specular_color = vec4(0.56, 0.48, 0.06,  1.0);
 
	// sunlight reflection and refraction
	vec3 reflectDir = reflect(lightdir, norm);
	float e = 1.0f / 1.34f;
	vec3 refractDir = refract(lightdir, norm, e);
	float fresnelBias = 0.3;
	float fresnelPower = 0.6;
	float fresnelScale = 1.0;
	float fresnel = fresnelBias + fresnelScale * pow(min(0.0, 1.0-dot(lightdir, norm)), fresnelPower);
	float spec = pow(max(dot(viewDir, reflectDir), 0.0), 64.0);

	// 这里不应该一样
	vec4 sunSpecular = vec4(fresnel) * specular_color * spec;
	vec4 diffuse = vec4(1.0 - fresnel) * diffuse_color * coeff;

	// skymap
	fresnelBias = 0.6;
	fresnelPower = 0.3;
	fresnelScale = 1.0;
	vec3 R = reflect(- viewDir, norm);
	fresnel = fresnelBias + fresnelScale * pow(min(0.0, 1.0-dot(-R, norm)), fresnelPower);
	vec4 skySpecular = texture(sky, R).rgba * vec4(fresnel);
	vec4 skyDiffuse = texture(sky, R).rgba * vec4(1.0f - fresnel);
    // compute sky texture
    float ambient_contribution  = 0.6;
    float diffuse_contribution  = 1.2;
    float specular_contribution = 3.0;
 
    fragColor =  specular_contribution * skySpecular * sunSpecular + ambient_color *  diffuse * ambient_contribution + diffuse_contribution * skyDiffuse;
	          //  +
				//+ diffuse_contribution * skyDiffuse;
//            ambient_color  * ambient_contribution  * c +
//            diffuse_color  * diffuse_contribution  * c * max(d, 0) +
//                    (facing ?
//            specular_color * specular_contribution * c * max(pow(dot(normal1, halfway_vector1), 120.0), 0.0) :
//            vec4(0.0, 0.0, 0.0, 0.0));
// 
    //fragColor = fragColor * (1.0-fog_factor) + vec4(0.25, 0.75, 0.65, 1.0) * (fog_factor);
 
}