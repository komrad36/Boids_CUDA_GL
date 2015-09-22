/*******************************************************************
*   CUDA_GL.cu
*   Boids_CUDA_GL
*	Kareem Omar
*
*	6/18/2015
*   This program is entirely my own work.
*******************************************************************/

// This module contains the entirety of the OpenGL, CUDA,
// and CUDA-GL interop for maximum performance. This is
// a showcase of high-performance computing and thus some
// readability sacrifices were made. Comments throughout
// explain rationale and basic optimization trajectory.

// CUDA kernel (first compiled by NVCC, then MSVC)

#include "CUDA_GL.h"

// Random number generator for Boid initial positions and velocities
std::random_device rd;
std::mt19937 gen(rd());
#ifdef NEIGHBORHOOD_MATRIX
std::uniform_real_distribution<float> position_random_dist(0.0f, BOID_STARTING_BOX);
std::uniform_real_distribution<float> velocity_random_dist(-fMAX_START_VEL, fMAX_START_VEL);
#else
std::uniform_real_distribution<float> position_random_dist(0.0f, fP_MAX);
#endif

// need static instance because callbacks don't understand member functions
CUDA_GL& gl = CUDA_GL::getInstance();

void CUDA_GL::cleanup() {
	cleanupCuda();
	glfwDestroyWindow(window);
	glfwTerminate();
}

GLFWwindow* CUDA_GL::initGL() {
	std::cout << "Initializing GLFW..." << std::flush;
	if (!glfwInit()) {
		std::cerr << "ERROR: Failed to initialize GLFW. Aborting." << std::endl;
		return nullptr;
	}
	std::cout << " done." << std::endl;

#ifdef FULLSCREEN
	GLFWmonitor* const primary = glfwGetPrimaryMonitor();
	const GLFWvidmode* const mode = glfwGetVideoMode(primary);

	glfwWindowHint(GLFW_RED_BITS, mode->redBits);
	glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
	glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
	glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
	window = glfwCreateWindow(mode->width, mode->height, "Boids", primary, nullptr);
	std::cout << "Fullscreen framebuffer created." << std::endl;

#else
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	window = glfwCreateWindow(width, height, "Boids", nullptr, nullptr);
	std::cout << "Window created." << std::endl;
#endif

	if (!window) {
		std::cerr << "ERROR: Failed to create window. Aborting." << std::endl;
		return nullptr;
	}

	glfwMakeContextCurrent(window);
#ifdef V_SYNC
	// wait 1 sync to flip buffer (V_SYNC)
	glfwSwapInterval(1);
	std::cout << "V_SYNC enabled." << std::endl;
#else
	// sync immediately for lower latency
	// may introduce tearing
	glfwSwapInterval(0);
	std::cout << "V_SYNC disabled." << std::endl;
#endif

	glfwGetFramebufferSize(window, &width, &height);
	std::cout << "Framebuffer reports dimensions " << width << "x" << height << '.' << std::endl;

	// register callbacks with OpenGL
	glfwSetKeyCallback(window, keyboardCallback);
	glfwSetMouseButtonCallback(window, mouseClickCallback);
	glfwSetWindowPosCallback(window, windowMoveCallback);
	std::cout << "Callbacks registered." << std::endl;

	std::cout << "Initializing GLEW..." << std::flush;
	glewInit();
	std::cout << " done." << std::endl;

	if (!glewIsSupported("GL_VERSION_2_0 ")) {
		std::cerr << "ERROR: Support for necessary OpenGL extensions missing. Aborting" << std::endl;
		return nullptr;
	}

	// set viewport and viewing modes
	glViewport(0, 0, width, height);
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);
	std::cout << "OpenGL viewport configured." << std::endl;

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// set proj matrix
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f);

	// create texture from pixel buffer
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	// bind texture
	glBindTexture(GL_TEXTURE_2D, textureID);
	std::cout << "OpenGL texture configured." << std::endl;

	GLenum err;
	if ((err = glGetError()) != GL_NO_ERROR) {
		do {
			std::cerr << "ERROR: OpenGL reports " << err << ". Aborting." << std::endl;
		} while ((err = glGetError()) != GL_NO_ERROR);
		return nullptr;
	}

	blanker_blocks = (width*height - 1) / THREADS_PER_BLOCK + 1;

	return window;
}

#ifdef NEIGHBORHOOD_MATRIX
__global__ void siftEvenCols(Boid* __restrict__ d_arr) {
	int boid = blockDim.x * blockIdx.x + threadIdx.x;
	if (boid >= NUM_BOIDS) return;
	int i = boid / SQRT_NUM_BOIDS;
	int j = boid % SQRT_NUM_BOIDS;
	if (j % 2 == 0) return;

	Boid* one = &d_arr[i*SQRT_NUM_BOIDS + j];
	Boid* two = &d_arr[i*SQRT_NUM_BOIDS + j - 1];

	// swap elements if required such that smaller element
	// (sorted by x-position) ends up on left
	if (one->x < two->x) {
		float x = one->x;
		float y = one->y;
		float vx = one->vx;
		float vy = one->vy;

		one->x = two->x;
		one->y = two->y;
		one->vx = two->vx;
		one->vy = two->vy;

		two->x = x;
		two->y = y;
		two->vx = vx;
		two->vy = vy;
	}
}

__global__ void siftOddCols(Boid* __restrict__ d_arr) {
	int boid = blockDim.x * blockIdx.x + threadIdx.x;
	if (boid >= NUM_BOIDS) return;
	int i = boid / SQRT_NUM_BOIDS;
	int j = boid % SQRT_NUM_BOIDS;
	if ((j % 2) || j == 0) return;

	Boid* one = &d_arr[i*SQRT_NUM_BOIDS + j];
	Boid* two = &d_arr[i*SQRT_NUM_BOIDS + j - 1];

	// swap elements if required such that smaller element
	// (sorted by x-position) ends up on left
	if (one->x < two->x) {
		float x = one->x;
		float y = one->y;
		float vx = one->vx;
		float vy = one->vy;

		one->x = two->x;
		one->y = two->y;
		one->vx = two->vx;
		one->vy = two->vy;

		two->x = x;
		two->y = y;
		two->vx = vx;
		two->vy = vy;
	}
}

__global__ void siftEvenRows(Boid* __restrict__ d_arr) {
	int boid = blockDim.x * blockIdx.x + threadIdx.x;
	if (boid >= NUM_BOIDS) return;
	int i = boid / SQRT_NUM_BOIDS;
	if (i % 2 == 0) return;
	int j = boid % SQRT_NUM_BOIDS;

	Boid* one = &d_arr[i*SQRT_NUM_BOIDS + j];
	Boid* two = &d_arr[(i - 1)*SQRT_NUM_BOIDS + j];

	// swap elements if required such that smaller element
	// (sorted by y-position) ends up on top
	if (one->y < two->y) {
		float x = one->x;
		float y = one->y;
		float vx = one->vx;
		float vy = one->vy;

		one->x = two->x;
		one->y = two->y;
		one->vx = two->vx;
		one->vy = two->vy;

		two->x = x;
		two->y = y;
		two->vx = vx;
		two->vy = vy;
	}
}

__global__ void siftOddRows(Boid* __restrict__ d_arr) {
	int boid = blockDim.x * blockIdx.x + threadIdx.x;
	if (boid >= NUM_BOIDS) return;
	int i = boid / SQRT_NUM_BOIDS;
	if ((i % 2) || i == 0) return;
	int j = boid % SQRT_NUM_BOIDS;

	Boid* one = &d_arr[i*SQRT_NUM_BOIDS + j];
	Boid* two = &d_arr[(i - 1)*SQRT_NUM_BOIDS + j];

	// swap elements if required such that smaller element
	// (sorted by y-position) ends up on top
	if (one->y < two->y) {
		float x = one->x;
		float y = one->y;
		float vx = one->vx;
		float vy = one->vy;

		one->x = two->x;
		one->y = two->y;
		one->vx = two->vx;
		one->vy = two->vy;

		two->x = x;
		two->y = y;
		two->vx = vx;
		two->vy = vy;
	}
}
#endif

void CUDA_GL::display() {
	if (not_paused) {
		// map cuda stream to access pbo
		cudaGraphicsMapResources(1, &cgr, cs);
		if (do_blank) {
			blanker << <blanker_blocks, THREADS_PER_BLOCK >> >(d_pbo, width*height);
		}
#ifdef NEIGHBORHOOD_MATRIX
		// sift NUM_SIFTS times to update matrix for any boids
		// which changed positions relative to other boids during previous frame
		for (int i = 0; i < NUM_SIFTS_PER_FRAME; ++i) {
			siftEvenCols << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(d_in);
			siftOddCols << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(d_in);
			siftEvenRows << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(d_in);
			siftOddRows << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(d_in);
		}
#endif

		// simulation kernel:
		kernel << <NUM_BLOCKS, THREADS_PER_BLOCK >> >(
			d_pbo,			// pixel buffer to write into
			d_in,			// boids are stored ping-pong style so that CUDA threads
								// can update boid info without syncing to other threads
			d_out,			// output (swaps with input each frame)
			mouse_x,
			mouse_y,
			width,
			height,

			// send 0 if no mouse buttons down, OR
			// send the repulsion multipler (attract vs. repel) times the weak factor if 1 button down, OR
			// send the repulsion multiplier times the strong factor if two or more buttons down
			(mouse_buttons_down) ? (repulsion_multiplier * ((mouse_buttons_down > 1) ? STRONG_DOWN_STRENGTH_FACTOR : WEAK_MOUSE_DOWN_STRENGTH_FACTOR)) : 0.0f,
			static_cast<int>(frame_time_sum / frame_times.size())); // moving average of time elapsed per frame (for physics propagation)

		// wait for all threads to complete
		cudaDeviceSynchronize();

		// free the graphics resources for rendering
		cudaGraphicsUnmapResources(1, &cgr, cs);

		// swap pointers of in and out arrays each frame so data is written back and forth (all on GPU)
		Boid* temp = d_in;
		d_in = d_out;
		d_out = temp;
	}

	// nothing to copy -  it's already on device!
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

	// lay simple quad on which we scribble the pixels - or rather, on
	// which they're already scribbled
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(0.0f, 0.0f, 0.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(0.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, 0.0f, 0.0f);
	glEnd();

	// flip to front buffer
	glfwSwapBuffers(window);

	// check for key/mouse input
	glfwPollEvents();

	// if in normal simulation, frame_times is full, so remove its oldest element
	// and update frame_time_sum accordingly (there's no need to sum it every frame;
	// just subtract from it when you remove a frame and add to it when you add a frame!
	// as long as the initial value is correct (0 in this case), this will maintain a correct
	// sum for use in averaging.
	//
	// if in first few frames of simulation, frame_times is still lengthening so skip this step.
	if (frame_times.size() >= NUM_FRAMES_TO_AVERAGE) {
		frame_time_sum -= frame_times.front();
		frame_times.erase(frame_times.begin());
	}

	// record current time
	std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();

	// push last frametime to frame_times
	frame_times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(now - frame_clock).count());

	// update running sum of all frames in frame_times
	frame_time_sum += frame_times.back();

	// if supposed to print FPS output...
#ifdef US_TO_PRINT_FPS
	// ...and if enough time has elapsed that it's time to print...
	long long us_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - FPS_clock).count();
	if (us_elapsed > US_TO_PRINT_FPS) {
		// ...print...
		std::cout << "FPS: " << fps_frames * us_elapsed / US_PER_SECOND << std::endl;
		// ...and update the clock to now.
		FPS_clock = now;
		fps_frames = 0;
	}

	++fps_frames;

#endif

	// save current time to be checked after render to determine
	// framedraw time
	frame_clock = std::chrono::high_resolution_clock::now();
}

void CUDA_GL::cleanupCuda() {
	cudaFree(d_in);
	cudaFree(d_out);
	std::cout << "CUDA memory freed." << std::endl;

	cudaGraphicsUnregisterResource(cgr);
	std::cout << "CUDA graphics resource unregistered." << std::endl;

	cudaStreamDestroy(cs);
	std::cout << "CUDA interop stream destroyed." << std::endl;

	cudaDeviceReset();
	std::cout << "CUDA device reset." << std::endl;

	if (textureID) {
		glDeleteTextures(1, &textureID);
		textureID = 0;
	}
	std::cout << "Texture destroyed." << std::endl;

	if (pbo) {
		glBindBuffer(GL_ARRAY_BUFFER, pbo);
		glDeleteBuffers(1, &pbo);
		pbo = 0;
	}
	std::cout << "Pixel buffer destroyed." << std::endl;
}

cudaError_t CUDA_GL::spawnBoids() {
	cudaError_t cudaStatus;

	// generate host-side random initial positions
#ifdef NEIGHBORHOOD_MATRIX
	// if in neighborhood matrix mode, we want the matrix
	// to start already sorted, so spawn each boid in a random
	// position within a box representing its portion of the
	// matrix. Because this slightly compromises the random
	// appearance at the start, add random velocity to help
	// compensate.
	int idx;
	for (int i = 0; i < SQRT_NUM_BOIDS; ++i) {
		for (int j = 0; j < SQRT_NUM_BOIDS; ++j) {
			idx = i*SQRT_NUM_BOIDS + j;
			h_in[idx].vx = velocity_random_dist(gen);
			h_in[idx].vy = velocity_random_dist(gen);
			h_in[idx].x = j*BOID_STARTING_BOX + position_random_dist(gen);
			h_in[idx].y = i*BOID_STARTING_BOX + position_random_dist(gen);
		}
	}
#else
	// if not in neighborhood matrix mode, we can simply
	// spawn the boids randomly anywhere on the screen
	for (auto&& boid : h_in) {
		boid.vx = boid.vy = 0.0f;
		boid.x = position_random_dist(gen);
		boid.y = position_random_dist(gen);
	}
#endif

	// transfer initial boids to GPU (the only time boids are transferred! They
	// reside fully in GPU global memory from now on)
	cudaStatus = cudaMemcpy(d_in, h_in, NUM_BOIDS * sizeof(Boid), cudaMemcpyHostToDevice);
	CHECK_FOR_CUDA_ERRORS;

	return cudaSuccess;

Error:
	cudaFree(d_in);
	cudaFree(d_out);

	return cudaStatus;
}

cudaError_t CUDA_GL::initCuda() {
	// each pixel is a uchar4 (RGBA)
	size_t size = sizeof(uchar4) * width * height;

	// generate buffer 1 and map pbo to it
	glGenBuffers(1, &pbo);
	std::cout << "Pixel buffer created." << std::endl;

	// attach to it
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	// allocate
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size, nullptr, GL_DYNAMIC_DRAW); //GL_DYNAMIC_COPY

	// enable textures
	glEnable(GL_TEXTURE_2D);

	// generate texture ID for buffer 1
	glGenTextures(1, &textureID);

	// attach to it
	glBindTexture(GL_TEXTURE_2D, textureID);

	// allocate. The last parameter is nullptr since we only
	// want to allocate memory, not initialize it
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, nullptr);

	// don't need mipmapping
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	std::cout << "Texture created." << std::endl;

	cudaError_t cudaStatus;

	// choose which GPU to run on
	cudaStatus = cudaSetDevice(CUDA_DEVICE);
	CHECK_FOR_CUDA_ERRORS;
	std::cout << "CUDA device " << CUDA_DEVICE << " selected." << std::endl;

	// register pbo as a CUDA graphics resource
	cudaGraphicsGLRegisterBuffer(&cgr, pbo, cudaGraphicsRegisterFlagsWriteDiscard);

	// create interop stream
	cudaStreamCreate(&cs);
	std::cout << "CUDA interop stream created." << std::endl;

	// map the resource into that stream
	cudaGraphicsMapResources(1, &cgr, cs);

	// get device ptr for that resource so CUDA can write into it
	cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&d_pbo), &size, cgr);
	std::cout << "CUDA graphics resource registered." << std::endl;

	// allocate GPU ping-pong Boid arrays
	cudaStatus = cudaMalloc(reinterpret_cast<void**>(&d_in), NUM_BOIDS * sizeof(Boid));
	CHECK_FOR_CUDA_ERRORS;
	cudaStatus = cudaMalloc(reinterpret_cast<void**>(&d_out), NUM_BOIDS * sizeof(Boid));
	CHECK_FOR_CUDA_ERRORS;
	std::cout << "CUDA memory allocated." << std::endl;

	cudaStatus = spawnBoids();
	CHECK_FOR_CUDA_ERRORS;
	std::cout << "Boids spawned." << std::endl;

	// initialize both clocks
	FPS_clock = frame_clock = std::chrono::high_resolution_clock::now();

	return cudaSuccess;

Error:
	cudaFree(d_in);
	cudaFree(d_out);

	return cudaStatus;
}

void keyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS || action == GLFW_REPEAT) {
		switch (key) {
		case GLFW_KEY_ESCAPE:
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		case GLFW_KEY_P:
			gl.not_paused = !gl.not_paused;
			// update frame clock (important when unpausing such that the next frame doesn't jump ahead
			// due to all the elapsed time)
			gl.frame_clock = std::chrono::high_resolution_clock::now();
			break;
		case GLFW_KEY_LEFT_CONTROL:
			gl.repulsion_multiplier *= -1.0f;
			break;
		case GLFW_KEY_SPACE:
			gl.do_blank = !gl.do_blank;
		}
	}
}

void mouseClickCallback(GLFWwindow* window, int button, int action, int mods) {
	gl.mouse_buttons_down += (action == GLFW_PRESS) - (action == GLFW_RELEASE);
	double x, y;
	glfwGetCursorPos(window, &x, &y);
	gl.mouse_x = static_cast<int>(x);
	gl.mouse_y = static_cast<int>(y);

	// only track cursor motion if the mouse is down;
	// no need to waste cycles doing so otherwise
	glfwSetCursorPosCallback(window, gl.mouse_buttons_down ? mouseMoveCallback : nullptr);
}

void mouseMoveCallback(GLFWwindow* window, double x, double y) {
	gl.mouse_x = static_cast<int>(x);
	gl.mouse_y = static_cast<int>(y);
}

void windowMoveCallback(GLFWwindow* window, int x, int y) {
	// there's no platform-indepenent way to get around the fact that some platforms
	// *cough cough* WINDOWS freeze display updates while a window is being dragged
	// (unless you use a secondary thread) but we can easily minimize damage by at
	// least resetting the clock while it's in motion so the only time lost is time
	// the user holds the window still after dragging. I intend this to be used
	// fullscreen anyway so I'm not adding a secondary thread for rendering.
	gl.frame_clock = std::chrono::high_resolution_clock::now();
}

__global__ void blanker(uchar4* __restrict__ const d_pbo, const int width_by_height) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < width_by_height) {
		// set RGB to (0, 0, 0)
		d_pbo[index].x = d_pbo[index].y = d_pbo[index].z = 0;
	}
}

#ifdef SCREEN_WRAP
// return shortest distance between coordinates c1 to c2,
// screenwrapping if necessary and preserving direction
// of travel information
__device__ float diff(const float c1, const float c2) {
	float direct_distance = c2 - c1;
	float wrap_distance = (c2 > c1) ? direct_distance - fP_MAX : direct_distance + fP_MAX;

	return fabsf(direct_distance) < fHALF_P_MAX ? direct_distance : wrap_distance;
}
#endif

__global__ void kernel(
	uchar4* __restrict__ const d_pbo,
	const Boid* __restrict__ const in,
	Boid* __restrict__ const out,
	const int mouse_x,
	const int mouse_y,
	const int width,
	const int height,
	const float repulsion_multiplier,
	const int mean_frame_time)
{

	int boid = blockDim.x * blockIdx.x + threadIdx.x;

	if (boid >= NUM_BOIDS) return;

	float diffx, diffy, factor;

	float time_factor = TICK_FACTOR*mean_frame_time;

	// bring boid's position and velocity into local memory
	float x = in[boid].x, y = in[boid].y;
	float Vx = in[boid].vx, Vy = in[boid].vy;

	float magV;

	// apply mouse attraction/repulsion rules
	// it's actually faster not to branch here
#ifdef SCREEN_WRAP
	diffx = diff(x, (mouse_x * P_MAX / width));
	diffy = diff(y, (mouse_y * P_MAX / height));
#else
	diffx = (mouse_x * P_MAX / width) - x;
	diffy = (mouse_y * P_MAX / height) - y;
#endif

	// CUDA optimization 101: rsqrtf is fast inverse square root - no more quake-style hackery, we have
	// hardware for this now
	//
	// fmaf is fused multiply-add.
	//
	// We update the velocity components by a factor proportional to time elapsed
	// and ratio of component distance to the cursor to the total distance to the cursor
	// for a natural-looking attraction model
	factor = repulsion_multiplier * rsqrtf(fmaf(diffx, diffx, fmaf(diffy, diffy, PREVENT_ZERO_RETURN)));
	Vx += time_factor * diffx * factor;
	Vy += time_factor * diffy * factor;

	int i, idx, ix, iy;
	// apply neighbor-related rules
	int neighbors = 0;
	//float neighbors = 0.0f;
	float CMsumX = 0.0f, CMsumY = 0.0f, REPsumX = 0.0f, REPsumY = 0.0f, ALsumX = 0.0f, ALsumY = 0.0f;
	// neighbors (includes own boid - optimization and prevents div-by-zero when no neighbors)
#ifdef NEIGHBORHOOD_MATRIX
	// get position in neighborhood matrix
	ix = boid / SQRT_NUM_BOIDS;
	iy = boid % SQRT_NUM_BOIDS;
	int j;
	// find boundaries of Moore radius box while not letting box go outside matrix
	int iterm = (ix < SQRT_NUM_BOIDS - MOORE_NEIGHBORHOOD_RADIUS) ? ix + MOORE_NEIGHBORHOOD_RADIUS : SQRT_NUM_BOIDS;
	int jterm = (iy < SQRT_NUM_BOIDS - MOORE_NEIGHBORHOOD_RADIUS) ? iy + MOORE_NEIGHBORHOOD_RADIUS : SQRT_NUM_BOIDS;

	// loop over Moore radius box
	for (i = (ix > MOORE_NEIGHBORHOOD_RADIUS) ? ix - MOORE_NEIGHBORHOOD_RADIUS : 0; i < iterm ; ++i) {
		for (j = (iy > MOORE_NEIGHBORHOOD_RADIUS) ? iy - MOORE_NEIGHBORHOOD_RADIUS : 0; j < jterm; ++j) {
			// reconstitute boid number (since we store boids as linear array)
			idx = i*SQRT_NUM_BOIDS + j;
#ifdef SCREEN_WRAP
			diffx = diff(x, in[idx].x);
			diffy = diff(y, in[idx].y);
#else
			diffx = in[idx].x - x;
			diffy = in[idx].y - y;
#endif
			// compute (diffx^2 + diffy^2) plus factor to prevent this from being 0
			// if same boid (faster not to branch)
			magV = fmaf(diffx, diffx, fmaf(diffy, diffy, PREVENT_ZERO_RETURN));

			// if this is a neighbor...
			if (magV < NEIGHBOR_DISTANCE_SQUARED) {
				// __fdividef is fast approximate float division
				factor = __fdividef(1.0f, magV);

				// update center of mass rule by distance and direction to neigbor
				CMsumX += diffx;

				// update repulsion rule by ratio of component distance to square of total distance to neighbor
				// for natural repulsion model
				REPsumX -= diffx * factor;

				// update alignment rule by component velocity of neighbor
				ALsumX += in[idx].vx;

				CMsumY += diffy;
				REPsumY -= diffy * factor;
				ALsumY += in[idx].vy;

				// keep track of total neighbor count for averaging these rule sums
				++neighbors;
			}
		}
	}
#else
	// sometimes when optimizing, particularly in CUDA, the fastest way
	// isn't pretty. a bit of code duplication here. explanations above in
	// the neighborhood matrix mode section.
	for (idx = 0; idx < NUM_BOIDS; ++idx) {
#ifdef SCREEN_WRAP
		diffx = diff(x, in[idx].x);
		diffy = diff(y, in[idx].y);
#else
		diffx = in[idx].x - x;
		diffy = in[idx].y - y;
#endif

		magV = fmaf(diffx, diffx, fmaf(diffy, diffy, PREVENT_ZERO_RETURN));

		if (magV < NEIGHBOR_DISTANCE_SQUARED) {
			factor = __fdividef(1.0f, magV);
			CMsumX += diffx;
			REPsumX -= diffx * factor;
			ALsumX += in[idx].vx;

			CMsumY += diffy;
			REPsumY -= diffy * factor;
			ALsumY += in[idx].vy;

			++neighbors;
		}
	}
#endif

#ifdef SCREEN_WRAP
	// okay, this is a fun one. We update the velocity component by the time factor multiplied by the center of mass average, which is the center of mass sum computed
	// in the loop above, divided by the number of neighbors.
	// We do the same with repulsion and alignment (there we must subtract our own velocity as it's the only rule affected by the fact that we chose to not
	// check whether the test boid is distinct (for speed), and thus count ourselves as a neighbor.
	Vx += fmaf(time_factor, __fdividef(CMsumX * CENTER_OF_MASS_STRENGTH_FACTOR, neighbors), fmaf(REPULSION_STRENGTH_FACTOR, REPsumX, __fdividef((ALsumX - in[boid].vx)*ALIGNMENT_STRENGTH_FACTOR, neighbors)));
	Vy += fmaf(time_factor, __fdividef(CMsumY * CENTER_OF_MASS_STRENGTH_FACTOR, neighbors), fmaf(REPULSION_STRENGTH_FACTOR, REPsumY, __fdividef((ALsumY - in[boid].vy)*ALIGNMENT_STRENGTH_FACTOR, neighbors)));
	// corrected for fact that own boid was counted as neighbor
#else
	// the same occurs with screenwrap off as the above description, with one change: now repulsion also includes
	// a term for repelling off the edges of the screen, if within range, inversely proportional to distance from edge
	Vx += fmaf(time_factor, __fdividef(CMsumX * CENTER_OF_MASS_STRENGTH_FACTOR, neighbors), fmaf(REPULSION_STRENGTH_FACTOR, REPsumX + EDGE_REPULSION_STRENGTH_FACTOR*(x < NEIGHBOR_DISTANCE)*(NEIGHBOR_DISTANCE - x) - EDGE_REPULSION_STRENGTH_FACTOR*(x > fP_MAX - NEIGHBOR_DISTANCE)*(x - (fP_MAX - NEIGHBOR_DISTANCE)), __fdividef((ALsumX - in[boid].vx)*ALIGNMENT_STRENGTH_FACTOR, neighbors)));
	Vy += fmaf(time_factor, __fdividef(CMsumY * CENTER_OF_MASS_STRENGTH_FACTOR, neighbors), fmaf(REPULSION_STRENGTH_FACTOR, REPsumY + EDGE_REPULSION_STRENGTH_FACTOR*(y < NEIGHBOR_DISTANCE)*(NEIGHBOR_DISTANCE - y) - EDGE_REPULSION_STRENGTH_FACTOR*(y > fP_MAX - NEIGHBOR_DISTANCE)*(y - (fP_MAX - NEIGHBOR_DISTANCE)), __fdividef((ALsumY - in[boid].vy)*ALIGNMENT_STRENGTH_FACTOR, neighbors)));
#endif

	// limit velocity if over V_LIM
	magV = Vx*Vx + Vy*Vy;
	factor = (magV > V_LIM_2) ? V_LIM * rsqrtf(magV) : 1.0f;
	Vx *= factor;
	Vy *= factor;

#ifdef SCREEN_WRAP
	// update position...
	x += Vx * time_factor;
	y += Vy * time_factor;
	// ...then screenwrap it and store the result both to
	// the local component and to the global out array
	out[boid].x = (x += fP_MAX*((x < 0.0f) - (x >= fP_MAX)));
	out[boid].y = (y += fP_MAX*((y < 0.0f) - (y >= fP_MAX)));
#else
	// if not screenwrapping,
	// adjust the sign of the velocity of any boid outside the box
	// so it's heading inside again in case that wasn't handled
	// by the repulsion force.
	//
	// do NOT just bring its position to some value like 0.0f
	// because then multiple boids would collide (share the exact
	// same position) and thus might move together in future
	// if their velocities also match (as they might well - reduced
	// to a zero or V_LIM equilibrium in a corner, say)...
	if (x < 0.0f) Vx = fabs(Vx);
	if (x >= fP_MAX) Vx = -fabs(Vx);
	if (y < 0.0f) Vy = fabs(Vy);
	if (y >= fP_MAX) Vy = -fabs(Vy);

	// ...and THEN update position so we move back inside
	// without looking too unnaturally bounded
	x += Vx * time_factor;
	y += Vy * time_factor;

	out[boid].x = x;
	out[boid].y = y;
#endif

	// store velocities back to global
	out[boid].vx = Vx;
	out[boid].vy = Vy;

	// convert position to screen frame
	ix = width * __fdividef(x, fP_MAX);
	iy = height * __fdividef(y, fP_MAX);

	// convert velocity to screen frame
	// Should be scaled by f_PMAX but is
	// not so we're off by a constant.
	// This is deliberate as the only reason
	// we care about velocity in screen frame is
	// to get angle of travel for color and line
	// draw, and the constant term will cancel when
	// we normalize the vector...
	Vx *= width;
	Vy *= height;

	// ...which is done here.
	magV = rsqrtf(fmaf(Vx, Vx, fmaf(Vy, Vy, PREVENT_ZERO_RETURN)));

	// + pi to make range 0-2pi
	float angle = atan2f(Vx *= magV, Vy *= magV) + fPI;

	// --- angle to RGB --- // (sorry, this is faster than function calls,
							//  even with CUDA's aggressive inlining)

	// mult by 3/pi, equivalent to dividing by (pi/3) (60 degrees)
	float section = angle * THREE_OVER_PI;
	i = static_cast<int>(section);
	float frac = section - i;

	uint8_t HSV_to_RGB[6] = { MAX_RGB, static_cast<uint8_t>(fMAX_RGB - fMAX_RGB*frac), 0, 0, static_cast<uint8_t>(fMAX_RGB*frac), MAX_RGB };

	// assuming max V and H, we can get RGB quickly by rotating them around, each separated by 60 degrees,
	// which we do quickly by using the 60 degree sector to index into the HSV_to_RGB array accordingly.
	uint8_t R = HSV_to_RGB[i];
	uint8_t G = HSV_to_RGB[(i + 4) % NUM_HSV_SECTORS];
	uint8_t B = HSV_to_RGB[(i + 2) % NUM_HSV_SECTORS];
	/////////////////////////


	// --- Bresenham's line draw --- //

	// use normalized velocity vector to
	// compute start and end points
	// of line in screen frame
	int x1 = ix - Vx*LINE_LENGTH;
	int x2 = ix + Vx*LINE_LENGTH;
	int y1 = iy - Vy*LINE_LENGTH;
	int y2 = iy + Vy*LINE_LENGTH;

	int dx, dy, sx, sy, D;

	// get signs of dx and dy
	sx = (x2 > x1) - (x2 < x1);
	sy = (y2 > y1) - (y2 < y1);

	// ensure positive
	dx = sx * (x2 - x1);
	dy = sy * (y2 - y1);

	// each octant needs slightly different treatment
	// in Bessengam's algorithm. If dy > dx we must
	// iterate over y and adjust x as needed along the way,
	// and vice versa for dx > dy.
	if (dy > dx) {
		D = 2 * dx - dy;
		for (i = 0; i < dy; ++i) {
			// pixel write //
#ifdef SCREEN_WRAP
			// wrap around if outside screen
			boid = (y1 + height*((y1 < 0) - (y1 >= height)))*width + x1 + width*((x1 < 0) - (x1 >= width));
#else
			// just cap at edges if outside screen
			boid = ((y1 >= 0) ? ((y1 < height) ? y1 : height - 1) : 0)*width + ((x1 >= 0) ? ((x1 < width) ? x1 : width - 1) : 0);
#endif
			d_pbo[boid].x = R;
			d_pbo[boid].y = G;
			d_pbo[boid].z = B;
			//////////////

			while (D >= 0) {
				D -= 2 * dy;
				x1 += sx;
			}

			D += 2 * dx;
			y1 += sy;
		}
	}
	else {
		D = 2 * dy - dx;
		for (i = 0; i < dx; ++i) {
			// pixel write
#ifdef SCREEN_WRAP
			boid = (y1 + height*((y1 < 0) - (y1 >= height)))*width + x1 + width*((x1 < 0) - (x1 >= width));
#else
			boid = ((y1 >= 0) ? ((y1 < height) ? y1 : height - 1) : 0)*width + ((x1 >= 0) ? ((x1 < width) ? x1 : width - 1) : 0);
#endif
			d_pbo[boid].x = R;
			d_pbo[boid].y = G;
			d_pbo[boid].z = B;
			//////////////

			while (D >= 0) {
				D -= 2 * dx;
				y1 += sy;
			}

			D += 2 * dy;
			x1 += sx;
		}
	}
	//////////////
}