/*************************************************************************
***************************    Application    ****************************
*************************************************************************/

#include <thread>
#include <functional>
#include "Application.h"

#include "display.hpp"
#include "runphysics.hpp"
#include "displayphysics.hpp"

using namespace ArcSim;
using namespace std::chrono_literals;

/*************************************************************************
***************************    Application    ****************************
*************************************************************************/
Application::Application() : m_IsRunning(false)
{

}

static bool s_IsRunning = false;

void idle()
{

	if (!s_IsRunning)
		return;

	sim_step();
	redisplay();
}

extern void zoom(bool in);

static void keyboard(unsigned char key, int x, int y)
{
	unsigned char esc = 27, space = ' ';
	if (key == esc)
	{
		exit(EXIT_SUCCESS);
	}
	else if (key == space)
	{
		::s_IsRunning = !::s_IsRunning;
	}
	else if (key == 's')
	{
		::s_IsRunning = !::s_IsRunning;
		idle();
		::s_IsRunning = !::s_IsRunning;
	}
	else if (key == 'z')
	{
		zoom(true);
	}
	else if (key == 'x')
	{
		zoom(false);
	}
}


void Application::RunSimulate(std::string jsonPath, std::string savingPath)
{
	init_physics(jsonPath, savingPath, false);

	GlutCallbacks cb;
	cb.idle = idle;
	cb.keyboard = keyboard;

	run_glut(cb);
}


Application::~Application()
{

}