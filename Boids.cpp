/*******************************************************************
*   Boids.cpp
*   Boids_CUDA_GL
*	Kareem Omar
*
*	6/18/2015
*   This program is entirely my own work.
*******************************************************************/

// Boids_CUDA_GL simulates bird ("boid") flocking behavior (or fish, crowds,
// etc.), resulting in emergent properties. Uses CUDA for computation on the
// GPU and outputs pixels directly to an OpenGL texture, showcasing CUDA-GL
// interoperability at extremely optimized performance. 17,000 boids
// at 100 fps even on last-generation hardware is typical. This simulation
// is primarily tuned for aesthetics, not physical accuracy, although
// careful selection of parameters can produce very flock-like emergent
// behavior. The color of each boid is determined by its direction of travel.
// Screen-wrapping, fullscreen, and spatial data structure storage for
// increased performance are all available as options and can be enabled and
// disabled as desired in params.h. Other parameters are also adjustable.
//
// Requires GLEW (static), GLFW (static), and CUDA (dynamic).
//
// Commands:
//	Space			-	toggle screen blanking
//	P				-	pause
//	LCTRL			-	toggle attraction/repulsion
//	ESC				-	quit
//	1 mouse button	-	weak attraction/repulsion
//	2 mouse buttons	-	strong attracton/repulsion

#include "Boids.h"

int main(int argc, char* argv[]) {
	// flush denormals to zero on Intel
	// to prevent unexpected performance drops in FPU
	// comment out if causing compatibility issues
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

	// enable memory leak checking in debug mode
#ifdef _DEBUG
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	// need static instance because callbacks don't understand member functions
	CUDA_GL& gl = CUDA_GL::getInstance();

	// init OpenGL
	std::cout << "Initializing OpenGL..." << std::endl;
	if (!gl.initGL()) return EXIT_FAILURE;

	// init CUDA and CUDA-GL interop
	std::cout << "Initializing CUDA-GL interop..." << std::endl;
	if (gl.initCuda() != cudaSuccess) return EXIT_FAILURE;

	// while ESC not pressed
	while (!glfwWindowShouldClose(gl.window))
		gl.display();

	std::cout << std::endl << "Shutting down..." << std::endl;

	gl.cleanup();
	return EXIT_SUCCESS;
}