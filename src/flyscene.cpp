#include "flyscene.hpp"
#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include <float.h>
#include <Tucano/shapes/box.hpp>

//value used to check 
constexpr float minCheck = 1e-8;
const Eigen::Vector4f backgroundColor = Eigen::Vector4f(0.9, 0.9, 0.9, 0);
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
  glClearColor(backgroundColor.x(), backgroundColor.y(), backgroundColor.z(), backgroundColor.w());
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // position the scene light at the last ray-tracing light source
  scene_light.resetViewMatrix();
  scene_light.viewMatrix()->translate(-lights.back());

  // render the scene using OpenGL and one light source
  

//phong.render(mesh, flycamera, scene_light);

  // render the ray and camera representation for ray debug
  ray.render(flycamera, scene_light);
  camerarep.render(flycamera, scene_light);



  Tucano::Shapes::Box myBox = Tucano::Shapes::Box(0.5, 0.5, 1.0);
  myBox.render(flycamera, scene_light);
                  
  //render reflections
  for (int i = 0; i < reflections.size(); i++) {
	  reflections[i].render(flycamera, scene_light);
  }

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
  Eigen::Vector3f origin = flycamera.getCenter();

  //calculate intersection point with scene(closest intersection found)
  inters_point intersectionstruc = intersection(origin, screen_pos);
  std::cout << "intersection after struct: " << intersectionstruc.point << std::endl;
  std::cout << "normal after struct: " << intersectionstruc.face.normal << std::endl;

  //if intersection is the infinite vector, the ray intersects with no triangle
  ray.setOriginOrientation(flycamera.getCenter(), dir);
  if (intersectionstruc.intersected) {
	  float height = (intersectionstruc.point - flycamera.getCenter()).norm();
	  ray.setSize(0.01, height);
  }

  //calculate reflection ray and draw it
  //reset reflections again
  reflections.clear();
  if (intersectionstruc.intersected) {
	  Eigen::Vector3f incoming = intersectionstruc.point - flycamera.getCenter();

	  Eigen::Vector3f normalized_normal = intersectionstruc.face.normal.normalized();

	  Eigen::Vector3f reflection = reflect(incoming, normalized_normal);
	  std::cout << "reflection: " << reflection << std::endl;
	  Tucano::Shapes::Cylinder reflected = Tucano::Shapes::Cylinder(0.01, 1.0, 16, 64);
	  reflected.resetModelMatrix();
	  reflected.setOriginOrientation(intersectionstruc.point, reflection.normalized());
	  reflected.setSize(0.01, 10);
	  reflections.push_back(reflected);
	  std::cout << reflections[0].getRadius() << std::endl;
	  std::cout << reflections[0].getHeight() << std::endl;
  }
 
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

	inters_point intersectionstruc = intersection(origin, dest);

	//Choose how many rays you want to shoot to light
	int totalRaysShot = 15;
	//Counter for how many rays reach the light
	int raysReachLight = 0;

	//Each iterations shoots a ray to light
	for (int i = 0; i < totalRaysShot; i++)
	{
		//Create two random floats between -0.075 and 0.075
		//The range will be hardcoded unless we find a way to get the radius of the light
		float randX = (rand() % 16) / 100 - 0.075;
		float randY = (rand() % 16) / 100 -0.075;
		float randZ = (rand() % 16) / 100 - 0.075;

		Eigen::Vector3f offset = Eigen::Vector3f(randX, randY, randZ);

		//Shoot a ray from hit point to light center shifted by the offset vector
		inters_point rayToLight = intersection(intersectionstruc.point, offset + lights.back());

		//See if ray reaches light. Increment counter if it does
		if (!rayToLight.intersected)
		{
			raysReachLight++;
		}
	}

	if (intersectionstruc.intersected == true) {
		//Multiply the rgb value of the pixel by the shadow ratio
		return (raysReachLight / totalRaysShot) * shade(0, MAX_REFLECT, intersectionstruc.point, intersectionstruc.point - origin, intersectionstruc.face);
	}
	//if miss then return background color
	return Eigen::Vector3f(backgroundColor.x(), backgroundColor.y(), backgroundColor.z());
}

//Calculates the direction of the refraction when the ray is inside the object and outside.
Eigen::Vector3f Flyscene::refractionV(Eigen::Vector3f& view, Eigen::Vector3f& normal, float& index) {
	
	float cos = clamp(view.dot(normal), -1.0f, 1.0f);
	float i = 1;
	float x = index;
	Eigen::Vector3f norm = normal;
	if (cos < 0) {
		cos = -cos;
	}
	else {
		std::swap(i, x);
		norm = -normal;
	}

	float y = i / x;

	float z = 1 - y * y * (1 - cos * cos);

	if (z < 0) {
		return Eigen::Vector3f(0.0, 0.0, 0.0);
	}
	else {
		return ((y * index) + (y * cos - sqrtf(z))) * norm;
	}


}

//Tucano::Shapes::Box myBox = Tucano::Shapes::Box(1.0, 1.0, 1.0);
//void renderBox(const Tucano::Camera& camera, const Tucano::Camera& light) {
//	myBox.render(camera, light);
//}

Flyscene::inters_point Flyscene::intersection(Eigen::Vector3f origin,
	Eigen::Vector3f dest) {

	Eigen::Vector3f intersectionv;
	std::vector<float> ts;
	std::vector<Eigen::Vector3f> directions;
	std::vector<Eigen::Vector3f> normals;
	std::vector<Tucano::Face> faces;
	for (int i = 0; i < mesh.getNumberOfFaces(); ++i) {
		Tucano::Face face = mesh.getFace(i);
		/*std::cout << i << std::endl;*/
		float alpha;
		float beta;

		Eigen::Vector3f directionV = (dest - origin);
		Eigen::Vector3f facenormal = face.normal.normalized();
		//float distance = pow((pow(directionV.x(), 2) + pow(directionV.y(), 2) + pow(directionV.z(), 2)), 0.5);
		directionV.normalize();
		Eigen::Vector4f homogeneous = mesh.getShapeModelMatrix() * mesh.getVertex(face.vertex_ids[0]);
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
			Eigen::Vector4f homogeneous = mesh.getShapeModelMatrix() * mesh.getVertex(face.vertex_ids[j]);
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
			/*std::cout << "Found intersection at" << std::endl << intersectionv << std::endl;
			std::cout << "distance: " << distance << " t: " << t << " alpha: " << alpha << " beta: " << beta << std::endl;*/
			ts.push_back(t);
			normals.push_back(facenormal);
			directions.push_back(directionV);
			faces.push_back(face);
		}
	}
	if (ts.size() == 0) {
		return Flyscene::inters_point{ false, Eigen::Vector3f(), Tucano::Face() };
	}
	else {
		//calc correct intersection point(closest to the camera)
		float min = *std::min_element(ts.begin(), ts.end());
		std::vector<float>::iterator indexit = std::find(ts.begin(), ts.end(), min);
		float index = indexit - ts.begin();
		//get material of corresponding face
		Eigen::Vector3f direction = directions[index];
		//get normalv
		/*std::cout << "normal before struct: " << normals[index] << std::endl;
		std::cout << "intersection point before struct: " << origin + min * direction << std::endl;*/
		Eigen::Vector3f point = origin + min * direction;
		Tucano::Face face = faces[index];
		return Flyscene::inters_point{ true, point, face };
	}
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

Eigen::Vector3f Flyscene::reflect(Eigen::Vector3f& incoming, Eigen::Vector3f& normal)
{
	/*std::cout << "incoming" << incoming << std::endl;
	std::cout << "normal" << normal << std::endl;*/

	normal.normalized();
	Eigen::Vector3f reflection = incoming - 2 * (incoming.dot(normal) * normal);
	return reflection;
}

Eigen::Vector3f Flyscene::shade(int level,int maxlevel, Eigen::Vector3f intersection, Eigen::Vector3f ray, Tucano::Face face) {
	if (level <= maxlevel) {
		return directColor(intersection,ray, face) + reflectColor(level, intersection, ray, face);
	}
	return directColor(intersection,ray, face);
}

Eigen::Vector3f Flyscene::directColor(Eigen::Vector3f p, Eigen::Vector3f ray, Tucano::Face face) {
	//ambient term
	Eigen::Vector3f lightIntensity = Eigen::Vector3f(1.0, 1.0, 1.0);
	Eigen::Vector3f ka = materials[face.material_id].getAmbient();
	Eigen::Vector3f ambient = lightIntensity.cwiseProduct(ka);
	//diffuse term
	Eigen::Vector3f normal = face.normal.normalized();
	Eigen::Vector3f lightDirection = (flycamera.getViewMatrix() * -lights[0]).normalized();
	Eigen::Vector3f kd = materials[face.material_id].getDiffuse();
	float diffuseDot = lightDirection.dot(normal);
	float diffuseBounded = (diffuseDot > 0.0) ? diffuseDot : 0.0;

	Eigen::Vector3f diffuse = lightIntensity.cwiseProduct(kd) * diffuseBounded;
	//specular term
	Eigen::Vector3f reflectV = reflect(ray, face.normal);
	inters_point newIntersection = intersection(p, reflectV - p);
	Eigen::Vector3f cameraV = flycamera.getCenter() - p;
	float specularDot = (reflectV.normalized()).dot(cameraV.normalized());
	Eigen::Vector3f ks = materials[face.material_id].getSpecular();
	float shininess = materials[face.material_id].getShininess();
	float specularBounded = (specularDot > 0.0) ? specularDot : 0.0;
	Eigen::Vector3f specular = lightIntensity.cwiseProduct(ks) * pow(specularBounded, shininess);

	return ambient + diffuse + specular;
}

Eigen::Vector3f Flyscene::reflectColor(int level, Eigen::Vector3f intersectionP, Eigen::Vector3f ray, Tucano::Face face) {
	Tucano::Material::Mtl current_material = materials[face.material_id];
	Eigen::Vector3f specular = current_material.getSpecular();

	Eigen::Vector3f reflectV = reflect(ray, face.normal);
	inters_point newIntersection = intersection(intersectionP, reflectV - intersectionP);
	//if reflection doesnt intersect then return background color
	if (newIntersection.intersected == false) {
		return Eigen::Vector3f(backgroundColor.x(), backgroundColor.y(), backgroundColor.z());
	}
	//check if material is reflective, if so then go calculate recursion
	if (specular.x() != 0 || specular.y() != 0 || specular.z() != 0) {
		return Flyscene::shade(++level, MAX_REFLECT, newIntersection.point, newIntersection.point - intersectionP, newIntersection.face);
	}
	return Eigen::Vector3f(backgroundColor.x(), backgroundColor.y(), backgroundColor.z());

}

class Structure {
	Eigen::Vector3f vertice;
	Tucano::Face faces;
public:
	std::vector<std::vector<Tucano::Face>> subdivide(Tucano::Mesh mesh);
};

std::vector<std::vector<Tucano::Face>> subdivide(Tucano::Mesh mesh) {
	std::vector<Tucano::Face> bb;
	std::vector<std::vector<Tucano::Face>> bb2;

	float totalFaces = 0;
	for (int i = 0; i < mesh.getNumberOfFaces; i++) {
		totalFaces++;
		bb.push_back(mesh.getFace(i));
	}
	float maxval = totalFaces;

	while (totalFaces > 10) {


		totalFaces /= 2;
		float temp = totalFaces;
		for (int i = 0; i < totalFaces; i++) {
			bb.push_back(mesh.getFace(i));
		}
		while (totalFaces < maxval) {
			bb2.push_back(bb);
			totalFaces += totalFaces;
		}
		totalFaces = temp;
		std::vector <Tucano::Face> ideal_bb_size = bb2.back;

	}
	return bb2;


