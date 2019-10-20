#version 330

/******** in variables from vertex shader (interpolated) ********/
in vec4 color;
in vec3 normal;
in vec4 vert;
in vec3 fragPosition;
/**********************************/

/******** output color of fragment ********/
out vec4 out_Color;
/**********************************/


/******** uniform variables ********/
uniform mat4 viewMatrix;
uniform mat4 lightViewMatrix;
/**********************************/

/***************** write your uniform variable code here ****************/
/******** uniform variables with material parameters ********/
uniform vec3 ka;
uniform vec3 kd;
uniform vec3 ks;
uniform vec3 lightProperty;
uniform float shininess;
/**********************************/

void main(void)
{

	vec3 norm_normal = normalize(normal);
	vec3 norm_lightDirection = normalize(fragPosition);
	float diff = max(dot(norm_normal, norm_lightDirection), 0.0);

	vec4 eyePosition = inverse(viewMatrix) * vec4(0.0, 0.0, 0.0, 1.0);
	vec3 norm_eyeWorld = normalize(eyePosition.xyz - vert.xyz);
	vec3 reflectDirection = reflect(-norm_lightDirection, norm_normal);
	vec3 norm_reflecD = normalize(reflectDirection);
	float spec = pow(max(dot(norm_eyeWorld, norm_reflecD), 0.0), shininess);

	vec3 ambient = surfacePropertyA;
	vec3 diffuse = surfacePropertyD * diff;
	vec3 specular = surfacePropertyS * spec;


	out_Color = vec4(lightProperty * (ambient + diffuse + specular), 1.0);
}