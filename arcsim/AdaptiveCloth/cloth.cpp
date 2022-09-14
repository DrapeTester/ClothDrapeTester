/*
  Copyright ©2013 The Regents of the University of California
  (Regents). All Rights Reserved. Permission to use, copy, modify, and
  distribute this software and its documentation for educational,
  research, and not-for-profit purposes, without fee and without a
  signed licensing agreement, is hereby granted, provided that the
  above copyright notice, this paragraph and the following two
  paragraphs appear in all copies, modifications, and
  distributions. Contact The Office of Technology Licensing, UC
  Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620,
  (510) 643-7201, for commercial licensing opportunities.

  IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT,
  INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
  LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
  DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY
  OF SUCH DAMAGE.

  REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING
  DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS
  IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
  UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "cloth.hpp"

void compute_masses(Cloth & cloth)
{
	for (int v = 0; v < cloth.mesh.verts.size(); v++)
		cloth.mesh.verts[v]->mass = 0.0f;

	for (int n = 0; n < cloth.mesh.nodes.size(); n++)
		cloth.mesh.nodes[n]->mass = 0.0f;

	for (int f = 0; f < cloth.mesh.faces.size(); f++)
	{
		Face * pFace = cloth.mesh.faces[f];

		pFace->mass = pFace->restArea * cloth.materials[pFace->label]->density;

		for (int v = 0; v < 3; v++)
		{
			pFace->v[v]->mass += pFace->mass / 3.0f;

			pFace->v[v]->node->mass += pFace->mass / 3.0f;
		}
	}
}