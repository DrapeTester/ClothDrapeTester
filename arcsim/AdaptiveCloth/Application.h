/*************************************************************************
***************************    Application    ****************************
*************************************************************************/
#pragma once

#include <string>

namespace ArcSim
{
	/*********************************************************************
	*************************    Application    **************************
	*********************************************************************/

	class Application
	{

	public:

		Application();
		~Application();

	public:

		void RunSimulate(std::string jsonPath, std::string savingPath = "");

	private:

		

	private:

		bool			m_IsRunning;
	};
}