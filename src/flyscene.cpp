#include "flyscene.hpp"
#include <GLFW/glfw3.h>

//value used to check 
constexpr float minCheck = 1e-8;

void Flyscene::initialize(int width, int height) {
  // initiliaze the Phong Shading effect for the Opengl Previewer
  phong.initialize();

  // set the camera's projection matrix
  flycamera.setPerspectiveMatrix(60.0, width / (float)height, 0.1f, 100.0f);
  flycamera.setViewport(Eigen::Vector2f((float)width, (float)height));

  // load the OBJ file and materials
  Tucano::MeshImporter::loadObjFile(mesh, materials,
                                    "resources/models/cube.obj");

  // normalize the model (scale to unit cube and center at origin)
  mesh.normalizeModelMatrix();

  // pass all the materials to the Phong Shader
  for (int i = 0; i < materials.size(); ++i)
    phong.addMaterial(materials[i]);

  // set the color and size of the sphere to represent the light sources
  // same sphere is used for all sources
  lightrep.setColor(Eigen::Vector4f(1.0, 1.0, 0.0, 1.0));
  lightrep.setSize(0.15);

  // create a first ray-tracing light source at some random position
  lights.push_back(Eigen::Vector3f(-1.0, 1.0, 1.0));

  // scale the camera representation (frustum) for the ray debug
  camerarep.shapeMatrix()->scale(0.2);

  // the debug ray is a cylinder, set the radius and length of the cylinder
  ray.setSize(0.005, 10.0);

  // craete a first debug ray pointing at the center of the screen
  createDebugRay(Eigen::Vector2f(width / 2.0, height / 2.0));

  glEnable(GL_DEPTH_TEST);

  // for (int i = 0; i<mesh.getNumberOfFaces(); ++i){
  //   Tucano::Face face = mesh.getFace(i);    
  //   for (int j =0; j<face.vertex_ids.size(); ++j){
  //     std::cout<<"vid "<<j<<" "<<face.vertex_ids[j]<<std::endl;
  //     std::cout<<"vertex "<<mesh.getVertex(face.vertex_ids[j]).transpose()<<std::endl;
  //     std::cout<<"normal "<<mesh.getNormal(face.vertex_ids[j]).transpose()<<std::endl;
  //   }
  //   std::cout<<"mat id "<<face.material_id<<std::endl<<std::endl;
  //   std::cout<<"face   normal "<<face.normal.transpose() << std::endl << std::endl;
  // }



}

void Flyscene::paintGL(void) {

  // update the camera view matrix with the last mouse interactions
  flycamera.updateViewMatrix();
  Eigen::Vector4f viewport = flycamera.getViewport();

  // clear the screen and set background color
  glClearColor(0.9, 0.9, 0.9, 0.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // position the scene light at the last ray-tracing light source
  scene_light.resetViewMatrix();
  scene_light.viewMatrix()->translate(-lights.back());

  // render the scene using OpenGL and one light source
  phong.render(mesh, flycamera, scene_light);

  // render the ray and camera representation for ray debug
  ray.render(flycamera, scene_light);
  camerarep.render(flycamera, scene_light);

  // render ray tracing light sources as yellow spheres
  for (int i = 0; i < lights.size(); ++i) {
    lightrep.resetModelMatrix();
    lightrep.modelMatrix()->translate(lights[i]);
    lightrep.render(flycamera, scene_light);
  }

  // render coordinate system at lower right corner
  flycamera.renderAtCorner();
}

void Flyscene::simulate(GLFWwindow *window) {
  // Update the camera.
  // NOTE(mickvangelderen): GLFW 3.2 has a problem on ubuntu where some key
  // events are repeated: https://github.com/glfw/glfw/issues/747. Sucks.
  float dx = (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS ? 1.0 : 0.0) -
             (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS ? 1.0 : 0.0);
  float dy = (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS
                  ? 1.0
                  : 0.0) -
             (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS
                  ? 1.0
                  : 0.0);
  float dz = (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS ? 1.0 : 0.0) -
             (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS ? 1.0 : 0.0);
  flycamera.translate(dx, dy, dz);
}

void Flyscene::createDebugRay(const Eigen::Vector2f &mouse_pos) {
  ray.resetModelMatrix();
  // from pixel position to world coordinates
  Eigen::Vector3f screen_pos = flycamera.screenToWorld(mouse_pos);

  // direction from camera center to click position
  Eigen::Vector3f dir = (screen_pos - flycamera.getCenter()).normalized();
  
  // calculate intersection
  Eigen::Vector3f origin = flycamera.getCenter();
  Eigen::Vector3f ray_intersect = intersection(origin, dir);
  
  // position and orient the cylinder representing the ray
  ray.setOriginOrientation(flycamera.getCenter(), -ray_intersect.normalized());
  //ray.setSize(0.01, ray_intersect.norm());

  // place the camera representation (frustum) on current camera location, 
  camerarep.resetModelMatrix();
  camerarep.setModelMatrix(flycamera.getViewMatrix().inverse());
}

void Flyscene::raytraceScene(int width, int height) {
  std::cout << "ray tracing ..." << std::endl;

  // if no width or height passed, use dimensions of current viewport
  Eigen::Vector2i image_size(width, height);
  if (width == 0 || height == 0) {
    image_size = flycamera.getViewportSize();
  }

  // create 2d vector to hold pixel colors and resize to match image size
  vector<vector<Eigen::Vector3f>> pixel_data;
  pixel_data.resize(image_size[1]);
  for (int i = 0; i < image_size[1]; ++i)
    pixel_data[i].resize(image_size[0]);

  // origin of the ray is always the camera center
  Eigen::Vector3f origin = flycamera.getCenter();
  Eigen::Vector3f screen_coords;

  // for every pixel shoot a ray from the origin through the pixel coords
  for (int j = 0; j < image_size[1]; ++j) {
    for (int i = 0; i < image_size[0]; ++i) {
      // create a ray from the camera passing through the pixel (i,j)
      screen_coords = flycamera.screenToWorld(Eigen::Vector2f(i, j));
      // launch raytracing for the given ray and write result to pixel data
      pixel_data[i][j] = traceRay(origin, screen_coords);
    }
  }

  // write the ray tracing result to a PPM image
  Tucano::ImageImporter::writePPMImage("result.ppm", pixel_data);
  std::cout << "ray tracing done! " << std::endl;
}


Eigen::Vector3f Flyscene::traceRay(Eigen::Vector3f& origin,
	Eigen::Vector3f& dest) {
	
	Eigen::Vector3f intersectionp = intersection(origin, dest);
	if (intersectionp != origin) {
		return Eigen::Vector3f(1, 0.5, 0);
	}
	return Eigen::Vector3f(0, 0, 0); 
}

Eigen::Vector3f Flyscene::reflect(Eigen::Vector3f& Inc, Eigen::Vector3f& Outc) {
	return Inc - (2 * Inc.dot(Outc) * Inc);


}

Eigen::Vector3f Flyscene::refraction(Eigen::Vector3f& Inc, Eigen::Vector3f& Outc, float& r) {
	float cos = clamp(Inc.dot(Outc), -1, 1);
}

Eigen::Vector3f Flyscene::intersection(Eigen::Vector3f& origin,
	Eigen::Vector3f& dest) {
	Eigen::Vector3f intersectionv;
	for (int i = 0; i < mesh.getNumberOfFaces(); ++i) {
		Tucano::Face face = mesh.getFace(i);
		float alpha;
		float beta;

		Eigen::Vector3f directionV = (dest - origin);
		Eigen::Vector3f facenormal = face.normal.normalized();
		//float distance = pow((pow(directionV.x(), 2) + pow(directionV.y(), 2) + pow(directionV.z(), 2)), 0.5);
		directionV.normalize();
		Eigen::Vector4f homogeneous = mesh.getVertex(face.vertex_ids[0]);
		Eigen::Matrix4f matrix = Eigen::Matrix4f(mesh.getShapeModelMatrix().matrix());
		//homogeneous = homogeneous.m * matrix;
		Eigen::Vector3f real = Eigen::Vector3f(homogeneous.x() / homogeneous.w(), homogeneous.y() / homogeneous.w(), homogeneous.z() / homogeneous.w());
		float distance = facenormal.dot(Eigen::Vector3f(homogeneous.x() / homogeneous.w(), homogeneous.y() / homogeneous.w(), homogeneous.z() / homogeneous.w()));
		float origin_normal = origin.dot(facenormal);		
		float direction_normal = directionV.dot(facenormal);

		//check whether ray is parallel to plane
		if (fabs(direction_normal) < minCheck || distance == 0) {
			continue;
		}

		float t = (distance - origin_normal) / direction_normal;
		if (t <= 0) {
			continue;
		}

		intersectionv = origin + t * directionV;

		//check whether intersection is inside triangle
		std::vector<Eigen::Vector3f> vectors;
		for (int j = 0; j < 3; j++) {
			Eigen::Vector4f homogeneous = mesh.getVertex(face.vertex_ids[j]);
			Eigen::Vector3f real = Eigen::Vector3f(homogeneous.x() / homogeneous.w(), homogeneous.y() / homogeneous.w(), homogeneous.z() / homogeneous.w());
			vectors.push_back(real);
		}

		float v1x = vectors[0][0];
		float v1y = vectors[0][1];
		float v2x = vectors[1][0];
		float v2y = vectors[1][1];
		float v3x = vectors[2][0];
		float v3y = vectors[2][1];

		//std::cout << v1x << " " << v1y << " " << v2x << " " << v2y << " " << v3x << " " << v3y << std::endl;

		//alpha = (v1x * (v3y - v1y) + (intersectionv.y() - v1y) * (v3x - v1x) - (intersectionv.x() * (v3y - v1y)))
		//	/ ((v2y - v1y) * (v3x - v1x) - (v2x - v1x) * (v3y - v1y));
		//beta = (intersectionv.y() - v1y - alpha * (v2y - v1y))
		//	/ (v3y - v1y);

		barycentric(intersectionv, vectors, alpha, beta);
		
		//if true then inside triangle

		if (alpha >= 0 && beta >= 0 && (alpha + beta) <= 1) {
			std::cout << "Found intersection at" << std::endl << intersectionv << std::endl;
			std::cout << "distance: " << distance << " t: " << t << " alpha: " << alpha << " beta: " << beta << std::endl;

			return intersectionv;
		}
	}
	return origin;
}

float Flyscene::clamp(float x, float low, float high) {
	return x < low ? low : x > high ? high : x;
}

void Flyscene::barycentric(Eigen::Vector3f p, std::vector<Eigen::Vector3f> vectors, float &alpha, float &beta) {
	Eigen::Vector3f v0 = vectors[1] - vectors[0];
	Eigen::Vector3f v1 = vectors[2] - vectors[0];
	Eigen::Vector3f v2 = p - vectors[0];

	float d00 = v0.dot(v0);
	float d01 = v0.dot(v1);
	float d11 = v1.dot(v1);
	float d20 = v2.dot(v0);
	float d21 = v2.dot(v1);
	float denom = d00 * d11 - d01 * d01;

	alpha = (d11 * d20 - d01 * d21) / denom;
	beta = (d00 * d21 - d01 * d20) / denom;
}
