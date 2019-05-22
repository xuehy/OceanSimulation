#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "loadShader.h"
#include "Tessendorf.h"
#include "genTexture.h"
#include "common.h"
using namespace std;

bool quit = false;
int Grid_Size = 64;
float field_size = 64;
// rotation angles and viewpoint
float pitch = glm::degrees(atan(50.0f/ 50.0f)), yaw = 90.0f;
glm::vec3 cameraPos = glm::vec3(0, 50, 50);
glm::vec3 cameraFront = glm::vec3(0, -50, -50);
glm::vec3 cameraUp = glm::vec3(0, 1, 0);
float lastX = 800, lastY = 600;
bool firstmouse = true;
float fov = 45.0f;
float deltaTime = 0.0f; // 当前帧与上一帧的时间差
float lastFrame = 0.0f; // 上一帧的时间
float elapsedTime = 0.0f;
void processInput(GLFWwindow* window)
{
	float cameraSpeed = 25.0f * deltaTime;
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		cameraPos += cameraSpeed * cameraFront;
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		cameraPos -= cameraSpeed * cameraFront;
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
	if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
	{
		// reset view
		yaw = 90;
		pitch = 0;
		cameraPos = glm::vec3(0, 0, 50);
		cameraFront = glm::vec3(0, 0, -50);
		fov = 45.0f;
	}
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
	{
		quit = true;
	}
}

void mouse_callback(GLFWwindow * window, double xpos, double ypos)
{
	if (firstmouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstmouse = false;
	}
	float xoffset = (float)xpos - lastX;
	float yoffset = lastY - (float)ypos; // 注意这里是相反的，因为y坐标是从底部往顶部依次增大的
	lastX = (float)xpos;
	lastY = (float)ypos;

	float sensitivity = 0.05f;
	xoffset *= sensitivity;
	yoffset *= sensitivity;
	yaw += xoffset;
	pitch += yoffset;
	if (pitch > 89.0f)
		pitch = 89.0f;
	if (pitch < -89.0f)
		pitch = -89.0f;
	glm::vec3 front;
	front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	front.y = sin(glm::radians(pitch));
	front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	cameraFront = -glm::normalize(front);
}

void scroll_callback(GLFWwindow * window, double xoffset, double yoffset)
{
	if (fov >= 1.0f && fov <= 45.0f)
		fov -= yoffset;
	if (fov <= 1.0f)
		fov = 1.0f;
	if (fov >= 45.0f)
		fov = 45.0f;
}
int main()
{
	if (!glfwInit()) {
		cerr << "Error: could not start GLFW3" << endl;
		return 1;
	}
	int width = 1280;
	int height = 960;
	// 抗锯齿
	glfwWindowHint(GLFW_SAMPLES, 4);
	glEnable(GL_MULTISAMPLE);
	GLFWwindow* window = glfwCreateWindow(width, height, "Hello OpenGL", NULL, NULL);
	if (!window) {
		cerr << "Error: could not open window with GLFW3" << endl;
		glfwTerminate();
		return 1;
	}
	glfwMakeContextCurrent(window);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback);
	glewExperimental = GL_TRUE;
	glewInit();
	//glEnable(GL_FRAMEBUFFER_SRGB);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	GLuint programID = LoadShaders("VertexShader.glsl", "FragmentShader.glsl");

	Ocean myOcean(Grid_Size, 0.0005f, glm::vec2(0.0f, 32.0f), field_size, false);
	// model view projection matrices and light position
	glm::mat4 Projection = glm::perspective(45.0f, (float)width / (float)height, 0.1f, 1000.0f);
	glm::mat4 View = glm::lookAt(
		cameraPos,
		cameraPos + cameraFront,
		cameraUp
	);
	glm::mat4 Model = glm::mat4(1.0f);
	glm::vec3 light_position;

	// sky renderer
	GLuint skyShader = LoadShaders("skyvertex.glsl", "skyfragment.glsl");
	vector<std::string> faces
	{
		"cloud/08/sky8_RT.jpg",
		"cloud/08/sky8_LF.jpg",
		"cloud/08/sky8_UP.jpg",
		"cloud/08/sky8_DN.jpg",
		"cloud/08/sky8_BK.jpg",
		"cloud/08/sky8_FR.jpg"
	};
	GLuint skyTexture = loadCubemap(faces);
	GLuint skyVAO, skyVBO;
	glGenVertexArrays(1, &skyVAO);
	glGenBuffers(1, &skyVBO);
	glBindVertexArray(skyVAO);
	glBindBuffer(GL_ARRAY_BUFFER, skyVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glUseProgram(skyShader);
	glUniform1i(glGetUniformLocation(skyShader, "skybox"), 0);
	glBindVertexArray(0);

	while (!quit)
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
		elapsedTime += deltaTime;
		processInput(window);
		View = glm::lookAt(
			cameraPos,
			cameraPos + cameraFront,
			cameraUp
		);
	
		
		light_position = glm::vec3(1000.0f, 1000.0f, -1000.0f);
		myOcean.render(elapsedTime * 10.0, light_position, Projection, View, Model, true);
		
		glDepthFunc(GL_LEQUAL);
		glUseProgram(skyShader);
		View = glm::mat4(glm::mat3(View)); // remove translation from the view matrix
		glUniformMatrix4fv(glGetUniformLocation(skyShader, "view"), 1, GL_FALSE, &View[0][0]);
		glUniformMatrix4fv(glGetUniformLocation(skyShader, "projection"), 1, GL_FALSE, &Projection[0][0]);

		// skybox cube
		glBindVertexArray(skyVAO);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_CUBE_MAP, skyTexture);
		glDrawArrays(GL_TRIANGLES, 0, 36);
		glBindVertexArray(0);
		glDepthFunc(GL_LESS); // set depth function back to default
		
		glfwPollEvents();
		glfwSwapBuffers(window);
	}
	myOcean.release();
	glfwTerminate();
	return 0;
}