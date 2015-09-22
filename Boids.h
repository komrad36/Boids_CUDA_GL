/*******************************************************************
*   Boids.h
*   Boids_CUDA_GL
*	Kareem Omar
*
*	6/18/2015
*   This program is entirely my own work.
*******************************************************************/

// Boids_CUDA_GL simulates bird ("boid"), fish, crowd, etc. flocking behavior,
// resulting in emergent properties. Uses CUDA for computation on the GPU
// and outputs pixels directly to an OpenGL texture with no memory transfer
// or CPU involvement whatsoever, showcasing CUDA-OpenGL
// interoperability at extremely optimized performance. 17,000 boids
// at 100 fps even on last-generation hardware is typical. This simulation
// is primarily tuned for aesthetics, not physical accuracy, although
// careful selection of parameters can produce very flock-like emergent
// behavior. The color of each boid is determined by its direction of travel.
// Screen-wrapping, fullscreen, and spatial data structure storage for increased
// performance are all available as options and can be enabled and disabled
// as desired in params.h. Simulation parameters are also available.
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

#ifndef BOIDS_H
#define BOIDS_H

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "params.h"
#include "CUDA_GL.h"

// in debug mode, set up infrastructure
// for memory leak checking
#ifdef _DEBUG

#ifndef DBG_NEW
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#define new DBG_NEW
#endif // DBG_NEW

#ifndef _CRTDBG_MAP_ALLOC
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif // _CRTDBG_MAP_ALLOC

#endif // _DEBUG

#endif