#version 150

/*********** in vertex attributes **********/
in vec4 in_Position;
in vec3 in_Normal;
in vec4 in_Color;

/********* going to fragment shader (will be interpolated) ********/
out vec4 color;
out vec3 normal;
out vec4 vert;
out vec3 fragPosition;

/************* uniform variables **********************/
uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform vec4 default_color;
uniform mat4 lightViewMatrix;

void main(void)
{

	vert = modelMatrix * in_Position;
	color = default_color;

    gl_Position = projectionMatrix * viewMatrix * modelMatrix * in_Position;
	fragPosition = vec3(lightViewMatrix * vec4(0.0, 0.0, 1.0, 0.0));
	normal = in_Normal;	

}