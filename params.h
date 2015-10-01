/*******************************************************************
*   params.h
*   Boids_CUDA_GL
*	Kareem Omar
*
*	6/18/2015
*   This program is entirely my own work.
*******************************************************************/

// This module contains user-configurable and other parameters.

#ifndef PARAMS_H
#define PARAMS_H

//################ User-configurable Parameters ################

// Recommended for performance and aesthetics
#define FULLSCREEN

// Not required; enable if tearing occurs
//				 disable if input latency occurs
//#define V_SYNC


// The most time-critical part of flocking simulation is the
// nearest-neighbor search, which is O(n^2). The rest of the simulation
// is O(n) with a small constant term - up to 98% of execution time is
// spent in the O(n^2) part. Obviously, improving this is critical to
// increasing performance. One option is the construction
// of a k-d tree for each frame, to be reused by all boids. However, a
// faster, though approximative, alternative takes advantage of the fact that
// boids do not change much from one frame to the next - a "neighborhood matrix"
// that, instead of being sorted every frame, is merely sifted a few times.
// This makes the critical code section O(r^2), where m is the Moore neighborhood
// radius, typically small compared to n! The speedups can be enormous - I have
// demonstrated one MILLION birds at 70 frames per second on an overclocked GTX 780m.
// HOWEVER - as the intent of this program is mainly aesthetic
// (hence the color being determined by direction of travel), and as
// properties which contribute to this, such as screen wrapping, strong
// mouse attraction, and excessively high rule strengths, all are
// detrimental to the performance of the neighborhood matrix approximation,
// I provide both options and typically do not actually use the matrix method.
// With careful tweaking a realstic simulation can be produced with it, but
// retaining the aesthetic quality due to the factors mentioned above
// requires high settings for number of sifts per frame and Moore
// neighborhood radius, thus reducing performance significantly while
// still not quite looking as nice as the exact mode. I thus default
// the application to the exact mode, but matrix mode can be enabled
// for experimentation here:
//
// NOTE: I highly recommend disabling screen wrap if neighborhood matrix
// is enabled, as birds disappearing from one edge and reappearing on the other
// violates the assumption of the matrix that the birds do not change much from
// one frame to the next, resulting in simulation inaccuracy.
//
//#define NEIGHBORHOOD_MATRIX

#define SCREEN_WRAP

// ---- Integer defines ----

// Square root of the number of boids simulated
#define		SQRT_NUM_BOIDS							(130)

// if FULL_SCREEN is not defined
#define		WINDOW_WIDTH							(3 * 1920 / 4)
#define		WINDOW_HEIGHT							(3 * 1080 / 4)

#define		CUDA_DEVICE								(0)

// if in neighborhood matrix mode, how many sifts (sorting passes) per frame?
// 1 would be sufficient for a carefully tuned simulation. For the aesthetic
// design of mine at the moment, 10 is recommended, although slow.
#define		NUM_SIFTS_PER_FRAME						(5)

// number of frames into the past to track with a moving average
// for determination of physics delta-t's
#define		NUM_FRAMES_TO_AVERAGE					(6)


// how often to print current FPS to console, or comment out altogether to disable
#define		US_TO_PRINT_FPS							(1000000)

// if in neighborhood matrix mode, radius of nearest neighbors
// to check and perform rules on (if actually within neighbor distance)
// This is the primary time sink for neighborhood radius mode. It's O(r^2).
// For a carefully tuned simulation, 6 is sufficient.
// However, for the aesthetic design of my sim at the moment, 20+ is
// recommended, although slow.
#define		MOORE_NEIGHBORHOOD_RADIUS				(19)

// CUDA threads per CUDA execution block. Empirically set to 128.
#define		THREADS_PER_BLOCK						(128)

// ---- Float defines ----

// length of lines representing boids' position and velocity
#define		LINE_LENGTH								(5.0f)

// if in neighborhood matrix mode, initial velocity
// Offered to improve random appearance since in
// neighborhood matrix mode, boids start in predefined boxes
// so there's less initial position randomness
#define		fMAX_START_VEL							(40.0f)

// speed limit of boids
#define		V_LIM									(200.0f)

// speed of physics
#define		TICK_FACTOR								(0.000005f)

// strength of alignment rule (boids try to fly in the same
// direction as their neighbors)
#define		ALIGNMENT_STRENGTH_FACTOR				(0.015f)

// strength of repulsion rule (boids repel from very
// nearby neighbors)
#define		REPULSION_STRENGTH_FACTOR				(55.0f)

// strength of edge repulsion rule, if screen wrapping is off
#define		EDGE_REPULSION_STRENGTH_FACTOR			(0.0002f)

// strength of center of mass rule (boids try to move toward
// their neighbors)
#define		CENTER_OF_MASS_STRENGTH_FACTOR			(0.6f)

// strength of one mouse button down mouse attaction/repulsion (boids fly
// in the direction of the mouse)
#define		WEAK_MOUSE_DOWN_STRENGTH_FACTOR			(100.0f)

// strength of two mouse button down mouse attaction/repulsion (boids fly
// in the direction of the mouse)
#define		STRONG_DOWN_STRENGTH_FACTOR				(6000.0f)

//##############################################################


#define	NUM_BOIDS					(SQRT_NUM_BOIDS*SQRT_NUM_BOIDS)
#define US_PER_SECOND				(1000000)
#define THREE_OVER_PI				(0.95492965855137f);
#define fPI							(3.14159265359f);

#define NUM_BLOCKS					((NUM_BOIDS - 1) / THREADS_PER_BLOCK + 1)

// square coordinate system max value (for screen independence)
#define P_MAX						(10000)
#define fP_MAX						(10000.0f)
#define fHALF_P_MAX					(5000.0f)

// for neighborhood matrix mode since we need the matrix to start sorted
#define BOID_STARTING_BOX			(P_MAX / SQRT_NUM_BOIDS)

// max distance to be considered a neighbor
#define NEIGHBOR_DISTANCE			(P_MAX / 13)
#define NEIGHBOR_DISTANCE_SQUARED	(NEIGHBOR_DISTANCE * NEIGHBOR_DISTANCE)

// for direction-of-travel to color conversion
#define NUM_HSV_SECTORS				(6)
#define MAX_RGB						(255)
#define fMAX_RGB					(255.0f)

// square of speed limit
#define	V_LIM_2						(V_LIM * V_LIM)

#define PREVENT_ZERO_RETURN			(0.000000001f)

#endif