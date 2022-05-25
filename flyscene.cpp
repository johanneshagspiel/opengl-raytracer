#include "flyscene.hpp"
#include <GLFW/glfw3.h>


void Flyscene::initialize(int width, int height) {
    // initiliaze the Phong Shading effect for the Opengl Previewer
    phong.initialize();


    // set the camera's projection matrix
    flycamera.setPerspectiveMatrix(60.0, width / (float) height, 0.1f, 100.0f);
    flycamera.setViewport(Eigen::Vector2f((float) width, (float) height));

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

    //set the color and the size of our debugOrbRep
    debugOrbRep.setColor(Eigen::Vector4f(1.0, 0.0, 0.0, 1.0));
    debugOrbRep.setSize(.08);

    // scale the camera representation (frustum) for the ray debug
    camerarep.shapeMatrix()->scale(0.2);

    // the debug ray is a cylinder, set the radius and length of the cylinder
    ray.setSize(0.005, 10.0);
	reflectionRay.setSize(0.005, 10.0);
	refractionRay.setSize(0.005, 10.0);

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
	reflectionRay.render(flycamera, scene_light);
	refractionRay.render(flycamera, scene_light);
    camerarep.render(flycamera, scene_light);

    // render ray tracing light sources as yellow spheres
    for (int i = 0; i < lights.size(); ++i) {
        lightrep.resetModelMatrix();
        lightrep.modelMatrix()->translate(lights[i]);
        lightrep.render(flycamera, scene_light);
    }

    //render the debugOrbRep's as red spheres
    for (int i = 0; i < debugOrbs.size(); i++) {
        debugOrbRep.resetModelMatrix();
        debugOrbRep.modelMatrix()->translate(debugOrbs[i]);
        debugOrbRep.render(flycamera, scene_light);
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


void Flyscene::raytraceScene(int width, int height) {
    std::cout << "ray tracing ..." << std::endl;

	std::vector<std::vector<Tucano::Face>> boxes = firstBox(mesh);
	std::vector<std::vector<Eigen::Vector3f>> boxbounds;
	for (int i = 0; i < boxes.size(); i++) {
		boxbounds.push_back(getBoxLimits(boxes[i], mesh));
	}

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
            pixel_data[i][j] = traceRay(origin, screen_coords, boxes, boxbounds);
        }
    }

    // write the ray tracing result to a PPM image
    Tucano::ImageImporter::writePPMImage("result.ppm", pixel_data);
    std::cout << "ray tracing done! " << std::endl;
}


//Intersection with plane
// src = slights
auto intersectPlane(Eigen::Vector3f start, Eigen::Vector3f to, Eigen::Vector3f normal, Eigen::Vector3f p) {
    
    float distance = p.dot(normal);
    Eigen::Vector3f point;
    struct result {
        bool inter;
        float t;
        Eigen::Vector3f point;
    };

	//std::cout << "Plane Intersect";

    if (to.dot(normal) == 0) {
        return result{false, 0, p};
    }

    float t = (distance - (start.dot(normal))) / (to.dot(normal));
    Eigen::Vector3f xyz = (start + t * to);

	//if intersection point and start is same doesnt count for shade
	if (xyz.x() == start.x() && xyz.y() == start.y() && xyz.z() == start.z()) {
		return result{ false, 0, p };
	}
    
	if (t < 0) {
		return result{ false, 0 , xyz};
	}
    return result{true, t, xyz};
}


//Intersection with triangle
//give normalized vectors
// src = https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates
auto intersectTriange(Eigen::Vector3f start, Eigen::Vector3f to, Tucano::Face face, Tucano::Mesh mesh) {

	const Eigen::Vector4f vec40 = (mesh.getShapeModelMatrix() * mesh.getVertex(face.vertex_ids[0]));
	const Eigen::Vector4f vec41 = (mesh.getShapeModelMatrix() * mesh.getVertex(face.vertex_ids[1]));
	const Eigen::Vector4f vec42 = (mesh.getShapeModelMatrix() * mesh.getVertex(face.vertex_ids[2]));

    Eigen::Vector3f facenormal = face.normal;

	Eigen::Vector3f v0 = (vec40 / vec40.w()).head<3>();
	Eigen::Vector3f v1 = (vec41 / vec41.w()).head<3>();
	Eigen::Vector3f v2 = (vec42 / vec42.w()).head<3>();
   
    struct result {
        bool inter;
        Tucano::Face face;
        Tucano::Mesh mesh;
        Eigen::Vector3f hit;
    };

	//std::cout << "Tiangle Intersect";

	auto planeinter = intersectPlane(start, to, facenormal, v0);

	if (planeinter.inter) {
		Eigen::Vector3f p_onPlane = planeinter.point;

		Eigen::Vector3f per;

		Eigen::Vector3f edge0 = v1 - v0;
		Eigen::Vector3f vp0 = p_onPlane - v0;
		per = edge0.cross(vp0);
		if (facenormal.dot(per) < 0) {
			return result{ false, face, mesh, p_onPlane };
		}


		Eigen::Vector3f  edge1 = v2 - v1;
		Eigen::Vector3f  vp1 = p_onPlane - v1;
		per = edge1.cross(vp1);
		if (facenormal.dot(per) < 0) {
			return result{ false, face, mesh, p_onPlane };
		}


		Eigen::Vector3f  edge2 = v0 - v2;
		Eigen::Vector3f  vp2 = p_onPlane - v2;
		per = edge2.cross(vp2);
		if (facenormal.dot(per) < 0) {
			return result{ false, face, mesh ,p_onPlane };
		}

		//std::cout << "TRIANGEL INTERSECTION FOUND " << std::endl;
		return result{ true, face, mesh, p_onPlane };

	}

	return result{ false, face, mesh, v0 };

}

//Object for git info
class HitData
{
	// Access specifier 
public:
	// Data Members
	Tucano::Face face;
	Eigen::Vector3f hit;
	
	// Member Functions() 
	Eigen::Vector3f gethit() {
		return hit;
	}
};

//gives the min and max of the box AND THE MEAN
std::vector<Eigen::Vector3f> Flyscene::getBoxLimits(std::vector<Tucano::Face> box, Tucano::Mesh mesh) {

    std::vector<Eigen::Vector3f> vecs;
    Eigen::Vector3f vec;
	Eigen::Vector4f vec4;

    if (box.size() == 0) {
        return vecs;
    }

    std::vector<GLuint> vecsofface;
	//std::cout << "sasa";

    for (int i = 0; i < box.size(); i++) {
        for (int a = 0; a < box[i].vertex_ids.size(); a++) {
            vecsofface.push_back(box[i].vertex_ids[a]);
        }
    }
	//std::cout << "sasa";

    float mean_x = 0;
    float mean_y = 0;
    float mean_z = 0;
	
	vec4 = mesh.getShapeModelMatrix() * mesh.getVertex(vecsofface[0]);
    vec = (vec4 / vec4.w()).head<3>();

    float min_x = vec.x();
    float max_x = vec.x();

    float min_y = vec.y();
    float max_y = vec.y();

    float min_z = vec.z();
    float max_z = vec.z();

    for (int i = 0; i < vecsofface.size(); i++) {
		Eigen::Vector4f v4 = mesh.getShapeModelMatrix() * mesh.getVertex(vecsofface[i]);
        Eigen::Vector3f v = (v4 / v4.w()).head<3>();;

        mean_x = mean_x + v.x();
        mean_y = mean_y + v.y();
        mean_z = mean_z + v.z();

        if (min_x > v.x()) {
            min_x = v.x();
        }
        if (max_x < v.x()) {
            max_x = v.x();
        }
        if (min_y > v.y()) {
            min_y = v.y();
        }
        if (max_y < v.y()) {
            max_y = v.y();
        }
        if (min_z > v.z()) {
            min_z = v.z();
        }
        if (max_z < v.z()) {
            max_z = v.z();
        }
    }
    mean_x = mean_x / vecsofface.size();
    mean_y = mean_y / vecsofface.size();
    mean_z = mean_z / vecsofface.size();
	//std::cout << "min " << Eigen::Vector3f(min_x, min_y, min_z) << std::endl;
	//std::cout << "max" << Eigen::Vector3f(max_x, max_y, max_z) << std::endl;

    vecs.push_back(Eigen::Vector3f(min_x, min_y, min_z));
    vecs.push_back(Eigen::Vector3f(max_x, max_y, max_z));
    vecs.push_back(Eigen::Vector3f(mean_x, mean_y, mean_z));
    return vecs;
}

//recursuve box
void Flyscene::createboxes(std::vector<Tucano::Face> box, Tucano::Mesh mesh, std::vector<std::vector<Tucano::Face>>& boxes) {
	// box is a vector with faces
	// boxes is a vector with boxes so vector -> vector(faces)
    std::vector<Tucano::Face> box1;
    std::vector<Tucano::Face> box2;

	//if box has less that 345 faces then push it to boxes
	// because if lower gets stucked
	if (box.size() < 345) {
		boxes.push_back(box);
		return;
	}

	//std::cout << "a" << std::endl;
	// box lim is the min max of box
	std::vector<Eigen::Vector3f> boxlim = Flyscene::getBoxLimits(box, mesh);

    float xdiff = boxlim[1].x() - boxlim[0].x();
    float ydiff = boxlim[1].y() - boxlim[0].y();
    float zdiff = boxlim[1].z() - boxlim[0].z();

    float cut;

	//std::cout << "red1 " << std::endl;
	//if x diff is the highest cut from there
    if (xdiff > ydiff && xdiff > zdiff) {
		//contains the mean
        cut = boxlim[2].x();

		//std::cout << " cut from x " << std::endl;
        for (int i = 0; i < box.size(); i++) {
			bool box1con = false;
			bool box2con = false;
			std::cout << box.size() << std::endl;
            for (int a = 0; a < box[i].vertex_ids.size(); a++) {
				Eigen::Vector4f v4 = mesh.getShapeModelMatrix() * mesh.getVertex(box[i].vertex_ids[a]);
				Eigen::Vector3f v = (v4 / v4.w()).head<3>();;
				if (v.x() < cut) {
					if (!box1con) {
						box1.push_back(box[i]);
						box1con = true;
					}
				}
				else {
					//changed this the same as the code above
					if (!box2con) {
						box2.push_back(box[i]);
						box2con = true;
					}
				}
            }
        }
		//if y dif is highest cut from y
    } else if (ydiff > xdiff && ydiff > zdiff) {
		//std::cout << " cut from y " << std::endl;
        cut = boxlim[2].y();
		std::cout << box.size() << std::endl;
        for (int i = 0; i < box.size(); i++) {
			bool box1con = false;
			bool box2con = false;

            for (int a = 0; a < box[i].vertex_ids.size(); a++) {
				Eigen::Vector4f v4 = mesh.getShapeModelMatrix() * mesh.getVertex(box[i].vertex_ids[a]);
				Eigen::Vector3f v = (v4 / v4.w()).head<3>();;
				if (v.y() < cut) {
					if (!box1con) {
						box1.push_back(box[i]);
						box1con = true;
					}
				}
				else {
					//changed this the same as the code above
					if (!box2con) {
						box2.push_back(box[i]);
						box2con = true;
					}
				}
            }
        }
		//if z dif is highest cut from z
    } else {
		//std::cout << " cut from z " << std::endl;
        cut = boxlim[2].z();
		std::cout << box.size() << std::endl;
        for (int i = 0; i < box.size(); i++) {
			bool box1con = false;
			bool box2con = false;
            for (int a = 0; a < box[i].vertex_ids.size(); a++) {
				Eigen::Vector4f v4 = mesh.getShapeModelMatrix() * mesh.getVertex(box[i].vertex_ids[a]);
				Eigen::Vector3f v = (v4 / v4.w()).head<3>();;
				if (v.z() < cut) {
					if (!box1con) {
						box1.push_back(box[i]);
						box1con = true;
					}
				}
				else {
					//changed this the same as the code above
					if (!box2con) {
						box2.push_back(box[i]);
						box2con = true;
					}
				}
            }
        }
    }
	//std::cout << "BOX1" << box1.size() << std::endl;
    createboxes(box1, mesh, boxes);
	//std::cout << "BOX2" << box2.size() << std::endl;
    createboxes(box2, mesh, boxes);

}

//first box
std::vector<std::vector<Tucano::Face>> Flyscene::firstBox(Tucano::Mesh mesh) {
    std::vector<Tucano::Face> box;
    std::vector<std::vector<Tucano::Face>> boxes;
    for (int i = 0; i < mesh.getNumberOfFaces(); i++) {
        box.push_back(mesh.getFace(i));
    }
	//std::cout << "big box" << box.size() << std::endl;
	//std::cout << "total faces" << mesh.getNumberOfFaces() << std::endl;
    createboxes(box, mesh, boxes);
    return boxes;
}

// checks intersection with bounding box
// src =  slights
std::vector<HitData> intersectBox(Eigen::Vector3f start, Eigen::Vector3f to, std::vector<std::vector<Tucano::Face>> boxes,
                  std::vector<std::vector<Eigen::Vector3f>> boxbounds, int inter_node, Tucano::Mesh mesh, std::vector<HitData>& hits) {
	//std::cout << "BOXessss " << boxes.size() << std::endl;
	//std::cout << "WE ARE " << inter_node << std::endl;
	

	if (boxes.size() <= inter_node) {
		//std::cout << " returned " << std::endl;
		std::cout << hits.size() << std::endl;
		return hits;
	}

	//std::cout << "BOX SIZE " << boxes[inter_node].size() << std::endl;

	if (boxes[inter_node].size() == 0) {
		return intersectBox(start, to, boxes, boxbounds, inter_node + 1, mesh, hits);
		
	}
	
	
	//std::cout << "red2 " << std::endl;
    float tmin_x = (boxbounds[inter_node][0].x() - start.x()) / to.x();
    float tmax_x = (boxbounds[inter_node][1].x() - start.x()) / to.x();

    float tmin_y = (boxbounds[inter_node][0].y() - start.y()) / to.y();
    float tmax_y = (boxbounds[inter_node][1].y() - start.y()) / to.y();

    float tmin_z = (boxbounds[inter_node][0].z() - start.z()) / to.z();
    float tmax_z = (boxbounds[inter_node][1].z() - start.z()) / to.z();

    float tin_x = std::min(tmin_x, tmax_x);
    float tout_x = std::max(tmin_x, tmax_x);

    float tin_y = std::min(tmin_y, tmax_y);
    float tout_y = std::max(tmin_y, tmax_y);

    float tin_z = std::min(tmin_z, tmax_z);
    float tout_z = std::max(tmin_z, tmax_z);

    float tin = std::max(tin_x, tin_y);
	tin = std::max(tin, tin_z);
    float tout = std::min(tout_x, tout_y);
	tout = std::min(tout, tout_z);

    if ((tin > tout) || tout < 0 ) {
		//std::cout << "box missed " << std::endl;
        if (boxes.size() <= inter_node) {
            return hits;
        }
        return intersectBox(start, to, boxes, boxbounds, inter_node + 1, mesh, hits);
    }
	//std::cout << "box hit " << std::endl;
    for (int i = 0; i < boxes[inter_node].size(); i++) {
        auto ans = intersectTriange(start, to, boxes[inter_node][i], mesh);
        if (ans.inter) {
			HitData hit;
			hit.face = ans.face;
			hit.hit = ans.hit;
            hits.push_back(hit);
        }
    }

    return intersectBox(start, to, boxes, boxbounds, inter_node + 1, mesh, hits);

}

//INTERSECT RETURNS A STRUCTURE 
// intersect(start, to, mesh).inter -> returns true if there is a intersection
// intersect(start, to, mesh).face -> returns the face that it hit if it didnt hit returns a random face
// intersect(start, to, mesh).hit -> returns the vec3f that it hit if it didnt hit returns a random hit point
//intersect of one vector to the universe
// FOR THE ACCELERATION YOU NEED TO FEED INTERSECT METHOD WITH THOSE CREATE THEM BEFORE LOOPING THEY ARE EXPENSIVE
//std::vector<std::vector<Tucano::Face>> boxes = firstBox(mesh);
//std::vector<std::vector<Eigen::Vector3f>> boxbounds;
//for (int i = 0; i < boxes.size(); i++) {
//boxbounds.push_back(getBoxLimits(boxes[i], mesh));
//}
auto intersect(Eigen::Vector3f start, Eigen::Vector3f to, Tucano::Mesh mesh, std::vector<std::vector<Tucano::Face>> boxes, std::vector<std::vector<Eigen::Vector3f>> boxbounds) {
    struct result {
        bool inter;
        Tucano::Face face;
        Eigen::Vector3f hit;
    };
	std::vector<HitData> lolz;
	std::vector<HitData> hits;

	/*
	for (int i = 0; i < mesh.getNumberOfFaces(); i++) {
		auto result = intersectTriange(start, to, mesh.getFace(i), mesh);
		if (result.inter) {
			HitData hit;
			hit.face = result.face;
			hit.hit = result.hit;
			hits.push_back(hit);
			hits.push_back(hit);
			std::cout << "we are at face " << i << std::endl;
		}
	}
	*/
	
    hits = intersectBox(start, to, boxes, boxbounds, 0, mesh, lolz);
	//std::cout << "hits size " <<hits.size()<< std::endl;
	if (hits.size() == 0) {
		return result{ false, mesh.getFace(0), Eigen::Vector3f(Eigen::Vector3f(0, 0, 0)) };
	}
	//std::cout << "red3 " << std::endl;
	// go through all the hits and get the smallest distance
	Eigen::Vector3f ahit = hits[0].gethit();
	float min_distance = sqrt((pow(start.x()-ahit.x(), 2) + pow(start.y() - ahit.y(), 2) + pow(start.z() - ahit.z(), 2)));
	float the_one = 0;
	for (int i = 0; i < hits.size(); i++) {
		ahit = hits[i].gethit();
		float d = sqrt((pow(start.x() - ahit.x(), 2) + pow(start.y() - ahit.y(), 2) + pow(start.z() - ahit.z(), 2)));
		if (d < min_distance) {
			the_one = i;
			min_distance = d;
		}
	}
    
	//std::cout << "the one " << the_one << std::endl;

    return result{true, hits[the_one].face, hits[the_one].gethit()};
}

// src = raytracing slight
Eigen::Vector3f refract(Eigen::Vector3f v, Eigen::Vector3f normal, float n1, float n2) {
	Eigen::Vector3f vnor = v.normalized();
	Eigen::Vector3f nnor = normal.normalized();
	float cosangle = vnor.dot(nnor);
	float snel = n1 / n2;
	float dot = v.dot(normal);

	return snel * (v - dot * normal) - normal * sqrt(1 - pow(snel,2)*(1-pow(dot,2)));
}

Eigen::Vector3f reflect(const Eigen::Vector3f I, const Eigen::Vector3f N) {
    return I - 2 * I.dot(N) * N;
}

auto Flyscene::hardShadow(Eigen::Vector3f hit, Tucano::Face face, Tucano::Mesh mesh, std::vector<Eigen::Vector3f> lights,
	std::vector<std::vector<Tucano::Face>> boxes, std::vector<std::vector<Eigen::Vector3f>> boxbounds) {
	float frac = 0;

	for (std::vector<Eigen::Vector3f>::iterator it = lights.begin(); it != lights.end(); ++it) {
		bool inter = intersect(*it, hit, mesh, boxes, boxbounds).inter;
		if (!inter) frac = frac += 1;
		///if (!inter) return true;
	}

	return frac / lights.size();
	//return false;
}

/* NOTE: whoever will implement hard shadow from a point, you need to check in the 2nd for loop, whether there is a face
 * between the light source and your current object. if so, you'll be in a shadow, otherwise, you're directly illuminated
Eigen::Vector3f directIllumination(const Eigen::Vector3f &I, const Eigen::Vector3f &N, std::vector<Eigen::Vector3f> lights,
                   Tucano::Mesh mesh) {
    std::vector<Eigen::Vector3f> lightvecs;
    for (int i = 0; i < lights.size(); i++) {
        Eigen::Vector3f current = (lights[i] - I);
        current = current.normalized();
        lightvecs.push_back(current);
    }
    for (int i = 0; i < lightvecs.size(); i++) {
        if (intersect(I, lightvecs[i], mesh).size() == 0) {

            float attenuation = 1 / (*dist);
            return lightColor * dot(N, L) * attenuation;

        }
    }
    if (!IsVisibile(I, L, dist)) return BLACK;

}
*/
// src = assignment 5
Eigen::Vector3f Flyscene::shade(int level, Eigen::Vector3f hit, Eigen::Vector3f from,Tucano::Face face, Tucano::Mesh mesh,
                      Tucano::Effects::PhongMaterial phong, std::vector<Eigen::Vector3f> lights,
                      std::vector<std::vector<Tucano::Face>> boxes,
                      std::vector<std::vector<Eigen::Vector3f>> boxbounds) {

	std::vector<Eigen::Vector3f> light_directions;
	std::vector<Eigen::Vector3f> reflected_lights;
	std::vector<Eigen::Vector3f> colors;
	
	Eigen::Vector3f normal3 = face.normal;
	/// 2) compute eye direction
	Eigen::Vector3f eye_vec3 = (-from).normalized();

	Eigen::Vector3f light_intensity = Eigen::Vector3f(1,1,1);
	
	
	Eigen::Vector3f ka = phong.getMaterial(face.material_id).getAmbient();
	Eigen::Vector3f kd = phong.getMaterial(face.material_id).getDiffuse();
	Eigen::Vector3f ks = phong.getMaterial(face.material_id).getSpecular();
	float n = materials[face.material_id].getShininess();

	for (int i = 0; i < lights.size(); i++) {
		Eigen::Vector3f color;
		/// 0) compute the light direction
		Eigen::Vector3f light_vec = (lights[i] - hit).normalized();
		light_directions.push_back(light_vec);

		/// 1) reflect light direction according to normal vector
		Eigen::Vector3f reflected_light = reflect(light_vec, normal3).normalized();
		reflected_lights.push_back(reflected_light);

		auto intersection = intersect(hit, light_vec, mesh, boxes, boxbounds);
		//std::cout << "INTERSECTTTTTTTT " << intersection.inter << std::endl;
		if (!intersection.inter) {
			/// 3) compute ambient, diffuse and specular components
			Eigen::Vector3f A = light_intensity.array() * ka.array();
			//std::cout << "AMBIENT " << A << std::endl;

			// normalized lenght = 1
			float cosss = normal3.dot(light_vec);
			Eigen::Vector3f D = light_intensity.array() * kd.array() * cosss;
			//std::cout << "DIFUSE " << D << std::endl;

			float coss = reflected_light.dot(eye_vec3);
			Eigen::Vector3f S = light_intensity.array() * ks.array() * pow(std::max(coss, 0.0f), n);
			//std::cout << "SPECULAR " << S << std::endl;
			// max because -shinenes doesnt make sense


			/// 4) compute final color using the Phong Model
			color = A + D + S;
			colors.push_back(color);
		}
		else {
			color = Eigen::Vector3f(0, 0, 0);
			colors.push_back(color);
		}
	}

	float sumx = 0;
	float sumy = 0;
	float sumz = 0;
	for (int n = 0; n < colors.size(); n++) {
		sumx = sumx + colors[n].x();
		sumy = sumy + colors[n].y();
		sumz = sumz + colors[n].z();
	}

	float frac = hardShadow(hit, face, mesh, lights, boxes, boxbounds);
	sumx = sumx * frac;
	sumy = sumy * frac;
	sumz = sumz * frac;

	return Eigen::Vector3f(sumx, sumy, sumz);

	// THIS PART IS UNKNOWN WHEN SURE PUT IT BEFORE RETURN
	// I am not even sgure if this part must be here or not
	// THIS PART I dont know but it should look something along the lines od this
	//need to implement with level

	//reflection
	Eigen::Vector3f reflection = reflect(from, normal3);
	
	shade(level - 1, hit, reflection, face, mesh, phong, lights, boxes, boxbounds);


	// refraction
	float air = 1.0;
	float material = materials[face.material_id].getOpticalDensity();

	Eigen::Vector3f refraction = refract(from, normal3, air, material);
	
	shade(level - 1, hit, refraction, face, mesh, phong, lights, boxes, boxbounds);
	

}

Eigen::Vector3f Flyscene::recursiveraytracing(int level, Eigen::Vector3f start, Eigen::Vector3f to, Tucano::Mesh mesh,
                                    Tucano::Effects::PhongMaterial phong, std::vector<Eigen::Vector3f> lights,
                                    std::vector<std::vector<Tucano::Face>> boxes,
                                    std::vector<std::vector<Eigen::Vector3f>> boxbounds) {
	std::cout << "recursive ray tracing" << std::endl;
	//return empty vector which is just supposed to be black
    auto intersection = intersect(start, to, mesh, boxes, boxbounds);
	std::cout << intersection.inter << std::endl;
    if (!intersection.inter) {
        return Eigen::Vector3f(1, 1, 1);
    }
	// if level is not 0 it should do the method agtain ?? we never do that here
    if (level == 0) {
        return start; //returns the coordinates for right now. whoever is working on shading and this function,
        // will determine if this is enough or they need to output color from this functions somehow.
    }
    return shade(level, intersection.hit, intersection.hit - start, intersection.face,
                 mesh, phong, lights, boxes, boxbounds); //either return just the color or after the shading
}


Eigen::Vector3f Flyscene::traceRay(Eigen::Vector3f &origin, Eigen::Vector3f &dest ,std::vector<std::vector<Tucano::Face>> &boxes, std::vector<std::vector<Eigen::Vector3f>> &boxbounds) {
    // just some fake random color per pixel until you implement your ray tracing
    // remember to return your RGB values as floats in the range [0, 1]!!!
    Eigen::Vector3f result = recursiveraytracing(1, origin, dest - origin, mesh, phong, lights, boxes,
                                                 boxbounds);   //bounce one more ray

    return result;
}


float distance(Eigen::Vector3f a, Eigen::Vector3f b) {
	float x = a.x() - b.x();
	float y = a.y() - b.y();
	float z = a.z() - b.z();

	return sqrt((pow(x,2)+ pow(y, 2)+ pow(z, 2)));
}

void Flyscene::createDebugRay(const Eigen::Vector2f &mouse_pos) {
    ray.resetModelMatrix();
	reflectionRay.resetModelMatrix();
	refractionRay.resetModelMatrix();
	
    // from pixel position to world coordinates
    Eigen::Vector3f screen_pos = flycamera.screenToWorld(mouse_pos);

    // direction from camera center to click position
    Eigen::Vector3f dir = (screen_pos - flycamera.getCenter()).normalized();

    // position and orient the cylinder representing the ray
    ray.setOriginOrientation(flycamera.getCenter(), dir);
	reflectionRay.setOriginOrientation(flycamera.getCenter(), dir);
	refractionRay.setOriginOrientation(flycamera.getCenter(), dir);


    // place the camera representation (frustum) on current camera location,
    camerarep.resetModelMatrix();
    camerarep.setModelMatrix(flycamera.getViewMatrix().inverse());

	std::cout << "debug " << std::endl;
	std::vector<std::vector<Tucano::Face>> boxes = firstBox(mesh);
	std::cout << "debug " << std::endl;
	std::vector<std::vector<Eigen::Vector3f>> boxbounds;
	for (int i = 0; i < boxes.size(); i++) {
		boxbounds.push_back(getBoxLimits(boxes[i], mesh));
	}

	std::cout << "debug " << std::endl;
	auto intersection = intersect(screen_pos, dir, mesh, boxes, boxbounds);

	std::cout << "inter " << intersection.inter << std::endl;
	if (intersection.inter) {
		//reflection
		std::cout << "hit data " << intersection.hit << std::endl;
		ray.setSize(0.005, distance(flycamera.getCenter(),intersection.hit));
		Eigen::Vector3f facenorm = intersection.face.normal.normalized();
		Eigen::Vector3f reflection = reflect(dir, facenorm);
		reflectionRay.setSize(0.005, 10);
		reflectionRay.setOriginOrientation(intersection.hit, reflection);
		Eigen::Vector3f color3 = shade(1, intersection.hit, intersection.hit - flycamera.getCenter(), intersection.face, mesh, phong, lights, boxes, boxbounds);
		Eigen::Vector4f color4 = Eigen::Vector4f(color3.x(), color3.y(), color3.z(), 1);
		reflectionRay.setColor(color4);
		std::cout << "COLOR " << color4 << std::endl;
		

		// src = your slights
		float air = 1.0;
		float material = materials[intersection.face.material_id].getOpticalDensity();

		std::cout << "MATERIAL "<< material << std::endl;
		
		Eigen::Vector3f refraction = refract(dir, facenorm, air, material);
		refractionRay.setSize(0.005, 10);
		refractionRay.setOriginOrientation(intersection.hit, refraction);
		refractionRay.setColor(Eigen::Vector4f(0, 1, 0, 0));
		
		std::cout << "arranged " << std::endl;
	}
	else {
		ray.setSize(0.005, 10.0);
	}

}