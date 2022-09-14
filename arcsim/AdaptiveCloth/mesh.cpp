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

#include "mesh.hpp"
#include "geometry.hpp"
#include "util.hpp"
#include <assert.h>
#include <cstdlib>

using namespace std;

template <typename T1, typename T2> void check(const T1 *p1, const T2 *p2,
											   const vector<T2*> &v2)
{
	if (p2 && find((T2*)p2, v2) == -1)
	{
		cout << p1 << "'s adjacent " << p2 << " is not accounted for" << endl;
		abort();
	}
}

template <typename T1, typename T2> void not_null(const T1 *p1, const T2 *p2)
{
	if (!p2)
	{
		cout << "adjacent to " << p1 << " is null " << p2 << endl;
		abort();
	}
}

template <typename T1, typename T2> void not_any_null(const T1 *p1, T2 *const*p2, int n)
{
	bool any_null = false;
	for (int i = 0; i < n; i++) if (!p2[i]) any_null = true;
	if (any_null)
	{
		cout << "adjacent to " << p1 << " one of these is null" << endl;
		for (int i = 0; i < n; i++) cout << p2[i] << endl;
		abort();
	}
}

template <typename T1, typename T2> void not_all_null(const T1 *p1, T2 *const*p2, int n)
{
	bool all_null = true;
	for (int i = 0; i < n; i++) if (p2[i]) all_null = false;
	if (all_null)
	{
		cout << "adjacent to " << p1 << " all of these are null" << endl;
		for (int i = 0; i < n; i++) cout << p2[i] << endl;
		abort();
	}
}

bool check_that_pointers_are_sane(const Mesh &mesh)
{
	for (int v = 0; v < mesh.verts.size(); v++)
	{
		const Vert *vert = mesh.verts[v];
		not_null(vert, vert->node);
		check(vert, vert->node, mesh.nodes);
		if (find((Vert*)vert, vert->node->verts) == -1)
		{
			cout << "vert " << vert << "'s node " << vert->node
				<< " doesn't contain it" << endl;
			abort();
		}
		for (int i = 0; i < vert->adjf.size(); i++)
			check(vert, vert->adjf[i], mesh.faces);
	}
	for (int n = 0; n < mesh.nodes.size(); n++)
	{
		const Node *node = mesh.nodes[n];
		for (int i = 0; i < node->verts.size(); i++)
			check(node, node->verts[i], mesh.verts);
		for (int i = 0; i < 2; i++)
			check(node, node->adje[i], mesh.edges);
	}
	for (int e = 0; e < mesh.edges.size(); e++)
	{
		const Edge *edge = mesh.edges[e];
		for (int i = 0; i < 2; i++)
			check(edge, edge->nodes[i], mesh.nodes);
		not_any_null(edge, edge->nodes, 2);
		for (int i = 0; i < 2; i++)
			check(edge, edge->adjf[i], mesh.faces);
		not_all_null(edge, edge->adjf, 2);
	}
	for (int f = 0; f < mesh.faces.size(); f++)
	{
		const Face *face = mesh.faces[f];
		for (int i = 0; i < 3; i++)
			check(face, face->v[i], mesh.verts);
		not_any_null(face, face->v, 3);
		for (int i = 0; i < 3; i++)
			check(face, face->adje[i], mesh.edges);
		not_any_null(face, face->adje, 3);
	}
	return true;
}

bool check_that_contents_are_sane(const Mesh &mesh)
{
	// // TODO
	// for (int v = 0; v < mesh.verts.size(); v++) {
	//     const Vert *vert = mesh.verts[v];
	//     if (!isfinite(norm2(vert->x + vert->v + vert->n) + vert->a)) {
	//         cout << "Vertex " << name(vert) << " is " << vert->x << " "
	//              << vert->v << " " << vert->n << " " << vert->a << endl;
	//         return false;
	//     }
	// }
	// for (int f = 0; f < mesh.faces.size(); f++) {
	//     const Face *face = mesh.faces[f];
	//     if (!isfinite(norm2(face->n) + face->a)) {
	//         cout << "Face " << name(face) << " is " << face->n << " "
	//              << face->a << endl;
	//         return false;
	//     }
	// }
	return true;
}

template<typename Type> Type learp(Type a, Type b, double ratio)
{
	return a * (1.0 - ratio) + b * ratio;
}
double dot_product(const std::vector<double> & x, const std::vector<double> & y)
{
	double result = 0.0;

	for (size_t i = 0; i < x.size(); i++)
	{
		result += x[i] * y[i];
	}

	return result;
}

//!		fitting function: y = p1 * cos(2x + p2) + p3.
void cos2_fitting(double & p1, double & p2, double & p3, const std::vector<double> & x, const std::vector<double> & y)
{
	std::vector<double> temp[3];

	temp[0].resize(x.size());
	temp[1].resize(x.size());
	temp[2].resize(x.size(), 1.0);

	for (size_t i = 0; i < x.size(); i++)
	{
		temp[0][i] =  cos(2.0 * x[i]);
		temp[1][i] = -sin(2.0 * x[i]);
	}

	Vec3 b;
	Mat3x3 A;

	for (int i = 0; i < 3; i++)
	{
		for (int j = i; j < 3; j++)
		{
			A(i, j) = dot_product(temp[i], temp[j]);
			A(j, i) = A(i, j);
		}

		b[i] = dot_product(temp[i], y);
	}

	if (det(A) != 0.0)
	{
		Vec3 r = A.inv() * b;
		double len = sqrt(r[0] * r[0] + r[1] * r[1]);

		p1 = len;
		p2 = atan2(r[1], r[0]);
		p3 = r[2];
	}
}

// Material space data

void compute_ms_data(Face* face)
{
	Vec2 A = face->v[0]->u;
	Vec2 B = face->v[1]->u;
	Vec2 C = face->v[2]->u;
	const Vec2 AB = B - A;
	const Vec2 BC = C - B;
	const Vec2 CA = A - C;

	face->Dm = Mat2x2(AB, -CA);
	face->invDm = face->Dm.inv();
	face->restArea = det(face->Dm) / 2;
	if (face->restArea == 0)
		face->invDm = Mat2x2(0);

	//!	for computing curveture.
	double a2 = dot(BC, BC);
	double b2 = dot(CA, CA);
	double c2 = dot(AB, AB);

	double dotB = -dot(AB, BC);
	double dotC = -dot(BC, CA);
	double dotA = -dot(CA, AB);

	double cotA = dotA / sqrt(b2 * c2 - dotA * dotA);
	double cotB = dotB / sqrt(a2 * c2 - dotB * dotB);
	double cotC = dotC / sqrt(a2 * b2 - dotC * dotC);

	face->restAngles[0] = acos(clamp(dotA / sqrt(b2) / sqrt(c2), -1.0, 1.0));
	face->restAngles[1] = acos(clamp(dotB / sqrt(a2) / sqrt(c2), -1.0, 1.0));
	face->restAngles[2] = acos(clamp(dotC / sqrt(a2) / sqrt(b2), -1.0, 1.0));
}

void compute_ms_data(Edge* edge)
{
	edge->l = 0;
	for (int s = 0; s < 2; s++)
		if (edge->adjf[s])
			edge->l += norm(edge_vert(edge, s, 0)->u - edge_vert(edge, s, 1)->u);
	if (edge->adjf[0] && edge->adjf[1])
		edge->l /= 2;
}

void compute_ms_data(Vert* vert)
{
	vert->area = 0;
	vert->restSumAngle = 0;

	const vector<Face*> &adjfs = vert->adjf;

	for (int i = 0; i < adjfs.size(); i++)
	{
		Face const* face = adjfs[i];
		vert->area += face->restArea / 3;

		int j = find(vert, face->v);
		vert->restSumAngle += face->restAngles[j];
	}
}

void compute_ms_data(Node* node)
{
	node->area = 0;
	node->restSumAngle = 0.0;
	node->principleAngle = 0.0;

	for (int v = 0; v < node->verts.size(); v++)
	{
		node->restSumAngle += node->verts[v]->restSumAngle;

		node->area += node->verts[v]->area;
	}
}

void compute_ms_data(Mesh &mesh)
{
	for (int f = 0; f < mesh.faces.size(); f++)
		compute_ms_data(mesh.faces[f]);
	for (int e = 0; e < mesh.edges.size(); e++)
		compute_ms_data(mesh.edges[e]);
	for (int v = 0; v < mesh.verts.size(); v++)
		compute_ms_data(mesh.verts[v]);
	for (int n = 0; n < mesh.nodes.size(); n++)
		compute_ms_data(mesh.nodes[n]);

	// now we need to recompute world-space data
	compute_ws_data(mesh);

	//double p1, p2, p3;
	//std::vector<double> x(100);
	//std::vector<double> y(100);
	//for (size_t i = 0; i < x.size(); i++)
	//{
	//	x[i] = 0.02 * i;
	//	y[i] = 2.3 * cos(2.0 * x[i] + 0.7) + 7.7;
	//}
	//cos2_fitting(p1, p2, p3, x, y);
}

// World-space data

void compute_ws_data(Face* face)
{
	const Vec3 & A = face->v[0]->node->x;
	const Vec3 & B = face->v[1]->node->x;
	const Vec3 & C = face->v[2]->node->x;
	
	const Vec3 AB = B - A;
	const Vec3 BC = C - B;
	const Vec3 CA = A - C;
	const Vec3 normal_vec = cross(AB, -CA);
	double triArea = norm(normal_vec) / 2.0;

	face->normal = normalize(normal_vec);
	face->tangent = normalize(AB * face->invDm(0, 0) - CA * face->invDm(1, 0));

	// Mat3x2 F = derivative(x0, x1, x2, face);
	// SVD<3,2> svd = singular_value_decomposition(F);
	// Mat3x2 Vt_ = 0;
	// for (int i = 0; i < 2; i++)
	//     for (int j = 0; j < 2; j++)
	//         Vt_(i,j) = svd.Vt(i,j);
	// face->R = svd.U*Vt_;
	// face->F = svd.Vt.t()*diag(svd.s)*svd.Vt;

	double a2 = dot(BC, BC);
	double b2 = dot(CA, CA);
	double c2 = dot(AB, AB);

	double dotB = -dot(AB, BC);
	double dotC = -dot(BC, CA);
	double dotA = -dot(CA, AB);

	double cotA = dotA / sqrt(b2 * c2 - dotA * dotA);
	double cotB = dotB / sqrt(a2 * c2 - dotB * dotB);
	double cotC = dotC / sqrt(a2 * b2 - dotC * dotC);

	double angleA = acos(clamp(dotA / sqrt(b2) / sqrt(c2), -1.0, 1.0));
	double angleB = acos(clamp(dotB / sqrt(a2) / sqrt(c2), -1.0, 1.0));
	double angleC = acos(clamp(dotC / sqrt(a2) / sqrt(b2), -1.0, 1.0));

	constexpr double half_pi = M_PI / 2.0;

	double mixedArea[3] = { triArea / 4.0, triArea / 4.0, triArea / 4.0 };

	if ((angleA < half_pi) && (angleB < half_pi) && (angleC < half_pi))
	{
		mixedArea[0] = (b2 * cotB + c2 * cotC) * (1.0f / 8.0f);
		mixedArea[1] = (c2 * cotC + a2 * cotA) * (1.0f / 8.0f);
		mixedArea[2] = (a2 * cotA + b2 * cotB) * (1.0f / 8.0f);
	}
	else if (angleA >= half_pi)		mixedArea[0] = triArea * 0.5f;
	else if (angleB >= half_pi)		mixedArea[1] = triArea * 0.5f;
	else							mixedArea[2] = triArea * 0.5f;

	face->meanCurvatureVectors[0] = -cotC * AB + cotB * CA;
	face->meanCurvatureVectors[1] = +cotC * AB - cotA * BC;
	face->meanCurvatureVectors[2] = -cotB * CA + cotA * BC;

	face->currMixAreas[0] = mixedArea[0];
	face->currMixAreas[1] = mixedArea[1];
	face->currMixAreas[2] = mixedArea[2];

	face->currAngles[0] = angleA;
	face->currAngles[1] = angleB;
	face->currAngles[2] = angleC;
}

void compute_ws_data(Edge *edge)
{
	edge->theta = dihedral_angle<WS>(edge);
}

//!	v0 & v1 e [-pi/2, pi/2).
double AngleLerp(double v0, double v1, double ratio)
{
	double v = v0;

	if (v1 - v0 > M_PI / 2)
		v = learp(v0, v1 - M_PI, ratio);
	else if (v0 - v1 > M_PI / 2)
		v = learp(v0, v1 + M_PI, ratio);
	else
		v = learp(v0, v1, ratio);

	if (v < -M_PI / 2)
		v += M_PI;
	else if (v > M_PI / 2)
		v -= M_PI;

	return v;
}

void compute_ws_data(Node* node)
{
	node->normal = Vec3(0);
	node->tangent = Vec3(0);
	double currSumAngle = 0.0;
	double currMixedArea = 0.0;
	Vec3 meanCurvatureVector = Vec3(0);

	for (int v = 0; v < node->verts.size(); v++)
	{
		const Vert *vert = node->verts[v];
		const vector<Face*> &adjfs = vert->adjf;

		for (int i = 0; i < adjfs.size(); i++)
		{
			Face const* face = adjfs[i];
			int j = find(vert, face->v);
			int j1 = (j + 1) % 3;
			int j2 = (j + 2) % 3;

			Vec3 e1 = face->v[j1]->node->x - node->x;
			Vec3 e2 = face->v[j2]->node->x - node->x;

			node->tangent += face->tangent * face->restArea;
			node->normal += cross(e1, e2) / (2 * norm2(e1)*norm2(e2));
			meanCurvatureVector += face->meanCurvatureVectors[j];
			currMixedArea += face->currMixAreas[j];
			currSumAngle += face->currAngles[j];
		}
	}

	node->normal = normalize(node->normal);
	node->tangent = normalize(node->tangent);
	meanCurvatureVector /= 2.0 * currMixedArea;
	double gaussianCurvature = (node->restSumAngle - currSumAngle) / currMixedArea;

	double s = (dot(node->normal, meanCurvatureVector) > 0.0) ? 1.0 : -1.0;
	double meanCurvature = s * norm(meanCurvatureVector) * 0.5;
	meanCurvatureVector = normalize(meanCurvatureVector);

	/////////////////////////////////////////////////////////////////////////////////////////////

	double kH = meanCurvature;
	double kS = gaussianCurvature;
	double delta = sqrt(max(0.0, kH * kH - kS));
	double k1 = kH + delta;
	double k2 = kH - delta;

	/////////////////////////////////////////////////////////////////////////////////////////////

	if (k1 != k2)
	{
		Vec3 posI = node->x;
		size_t numEdges = node->adje.size();
		std::vector<double>		x(numEdges);
		std::vector<double>		y(numEdges);

		for (size_t k = 0; k < numEdges; k++)
		{
			const Edge * edge = node->adje[k];

			int j = find(node, edge->nodes);
			j = (j + 1) % 2;

			Vec3 posJ = edge->nodes[j]->x;

			Vec3 dir = posI - posJ;

			y[k] = 2.0f * (dot(dir, meanCurvatureVector)) / norm2(dir);

			dir -= dot(dir, meanCurvatureVector) * meanCurvatureVector;
			dir = normalize(dir);

			Vec3 Ndir = -cross(node->tangent, dir);
			float sinAngle = norm(Ndir);
			if (dot(Ndir, meanCurvatureVector) < 0.0)
				sinAngle = -sinAngle;
			float cosAngle = dot(node->tangent, dir);
			x[k] = atan2(sinAngle, cosAngle);
		}

		double p1 = 0.0, p2 = 0.0, p3 = 0.0;
		cos2_fitting(p1, p2, p3, x, y);

		double stableAngle = p2 / 2.0;
		stableAngle = learp(0.0, stableAngle, min(abs(10.0 * meanCurvature), 1.0));
		node->principleAngle = AngleLerp(node->principleAngle, stableAngle, 0.999);
	}
	/////////////////////////////////////////////////////////////////////////////////////////////
}

void compute_ws_data(Mesh &mesh)
{
	for (int f = 0; f < mesh.faces.size(); f++)
		compute_ws_data(mesh.faces[f]);
	for (int e = 0; e < mesh.edges.size(); e++)
		compute_ws_data(mesh.edges[e]);
	for (int n = 0; n < mesh.nodes.size(); n++)
		compute_ws_data(mesh.nodes[n]);
}

// Mesh operations

template <> const vector<Vert*> &get(const Mesh &mesh) { return mesh.verts; }
template <> const vector<Node*> &get(const Mesh &mesh) { return mesh.nodes; }
template <> const vector<Edge*> &get(const Mesh &mesh) { return mesh.edges; }
template <> const vector<Face*> &get(const Mesh &mesh) { return mesh.faces; }

Edge *get_edge(const Node *n0, const Node *n1)
{
	for (int e = 0; e < n0->adje.size(); e++)
	{
		Edge *edge = n0->adje[e];
		if (edge->nodes[0] == n1 || edge->nodes[1] == n1)
			return edge;
	}
	return NULL;
}

Vert *edge_vert(const Edge *edge, int side, int i)
{
	Face *face = (Face*)edge->adjf[side];
	if (!face)
		return NULL;
	for (int j = 0; j < 3; j++)
		if (face->v[j]->node == edge->nodes[i])
			return face->v[j];
	return NULL;
}

Vert *edge_opp_vert(const Edge *edge, int side)
{
	Face *face = (Face*)edge->adjf[side];
	if (!face)
		return NULL;
	for (int j = 0; j < 3; j++)
		if (face->v[j]->node == edge->nodes[side])
			return face->v[PREV(j)];
	return NULL;
}

void connect(Vert *vert, Node *node)
{
	vert->node = node;
	include(vert, node->verts);
}

void Mesh::add(Vert *vert)
{
	verts.push_back(vert);
	vert->node = NULL;
	vert->adjf.clear();
	vert->index = verts.size() - 1;
}

void Mesh::remove(Vert* vert)
{
	if (!vert->adjf.empty())
	{
		cout << "Error: can't delete vert " << vert << " as it still has "
			<< vert->adjf.size() << " faces attached to it." << endl;
		return;
	}
	exclude(vert, verts);
}

void Mesh::add(Node *node)
{
	nodes.push_back(node);
	node->preserve = false;
	node->index = nodes.size() - 1;
	node->adje.clear();
	for (int v = 0; v < node->verts.size(); v++)
		node->verts[v]->node = node;
}

void Mesh::remove(Node* node)
{
	if (!node->adje.empty())
	{
		cout << "Error: can't delete node " << node << " as it still has "
			<< node->adje.size() << " edges attached to it." << endl;
		return;
	}
	exclude(node, nodes);
}

void Mesh::add(Edge *edge)
{
	edges.push_back(edge);
	edge->adjf[0] = edge->adjf[1] = NULL;
	edge->index = edges.size() - 1;
	include(edge, edge->nodes[0]->adje);
	include(edge, edge->nodes[1]->adje);
}

void Mesh::remove(Edge *edge)
{
	if (edge->adjf[0] || edge->adjf[1])
	{
		cout << "Error: can't delete edge " << edge
			<< " as it still has a face attached to it." << endl;
		return;
	}
	exclude(edge, edges);
	exclude(edge, edge->nodes[0]->adje);
	exclude(edge, edge->nodes[1]->adje);
}

void add_edges_if_needed(Mesh &mesh, const Face *face)
{
	for (int i = 0; i < 3; i++)
	{
		Node *n0 = face->v[i]->node, *n1 = face->v[NEXT(i)]->node;
		if (get_edge(n0, n1) == NULL)
			mesh.add(new Edge(n0, n1));
	}
}

void Mesh::add(Face *face)
{
	faces.push_back(face);
	face->index = faces.size() - 1;
	// adjacency
	add_edges_if_needed(*this, face);
	for (int i = 0; i < 3; i++)
	{
		Vert *v0 = face->v[NEXT(i)], *v1 = face->v[PREV(i)];
		include(face, v0->adjf);
		Edge *e = get_edge(v0->node, v1->node);
		face->adje[i] = e;
		int side = e->nodes[0] == v0->node ? 0 : 1;
		e->adjf[side] = face;
	}
}

void Mesh::remove(Face* face)
{
	exclude(face, faces);
	// adjacency
	for (int i = 0; i < 3; i++)
	{
		Vert *v0 = face->v[NEXT(i)];
		exclude(face, v0->adjf);
		Edge *e = face->adje[i];
		int side = e->nodes[0] == v0->node ? 0 : 1;
		e->adjf[side] = NULL;
	}
}

void update_indices(Mesh &mesh)
{
	for (int v = 0; v < mesh.verts.size(); v++)
		mesh.verts[v]->index = v;
	for (int f = 0; f < mesh.faces.size(); f++)
		mesh.faces[f]->index = f;
	for (int n = 0; n < mesh.nodes.size(); n++)
		mesh.nodes[n]->index = n;
	for (int e = 0; e < mesh.edges.size(); e++)
		mesh.edges[e]->index = e;
}

void mark_nodes_to_preserve(Mesh &mesh)
{
	for (int n = 0; n < mesh.nodes.size(); n++)
	{
		Node *node = mesh.nodes[n];
		if (is_seam_or_boundary(node) || node->label)
			node->preserve = true;
	}
	for (int e = 0; e < mesh.edges.size(); e++)
	{
		Edge *edge = mesh.edges[e];
		if (edge->label)
		{
			edge->nodes[0]->preserve = true;
			edge->nodes[1]->preserve = true;
		}
	}
}

void apply_transformation_onto(const Mesh &start_state, Mesh &onto,
							   const Transformation &tr)
{
	for (int n = 0; n < onto.nodes.size(); n++)
		onto.nodes[n]->x = tr.apply(start_state.nodes[n]->x);
	compute_ws_data(onto);
}

void apply_transformation(Mesh& mesh, const Transformation& tr)
{
	apply_transformation_onto(mesh, mesh, tr);
}

void update_x0(Mesh &mesh)
{
	for (int n = 0; n < mesh.nodes.size(); n++)
		mesh.nodes[n]->x0 = mesh.nodes[n]->x;
}

Mesh deep_copy(const Mesh &mesh0)
{
	Mesh mesh1;
	for (int v = 0; v < mesh0.verts.size(); v++)
	{
		const Vert *vert0 = mesh0.verts[v];
		Vert *vert1 = new Vert(vert0->u, vert0->label);
		mesh1.add(vert1);
	}
	for (int n = 0; n < mesh0.nodes.size(); n++)
	{
		const Node *node0 = mesh0.nodes[n];
		Node *node1 = new Node(node0->x, node0->v, node0->label);
		node1->preserve = node0->preserve;
		node1->verts.resize(node0->verts.size());
		for (int v = 0; v < node0->verts.size(); v++)
			node1->verts[v] = mesh1.verts[node0->verts[v]->index];
		mesh1.add(node1);
	}
	for (int e = 0; e < mesh0.edges.size(); e++)
	{
		const Edge *edge0 = mesh0.edges[e];
		Edge *edge1 = new Edge(mesh1.nodes[edge0->nodes[0]->index],
							   mesh1.nodes[edge0->nodes[1]->index],
							   edge0->label);
		mesh1.add(edge1);
	}
	for (int f = 0; f < mesh0.faces.size(); f++)
	{
		const Face *face0 = mesh0.faces[f];
		Face *face1 = new Face(mesh1.verts[face0->v[0]->index],
							   mesh1.verts[face0->v[1]->index],
							   mesh1.verts[face0->v[2]->index],
							   face0->label);
		mesh1.add(face1);
	}
	compute_ms_data(mesh1);
	return mesh1;
}

void delete_mesh(Mesh &mesh)
{
	for (int v = 0; v < mesh.verts.size(); v++)
		delete mesh.verts[v];
	for (int n = 0; n < mesh.nodes.size(); n++)
		delete mesh.nodes[n];
	for (int e = 0; e < mesh.edges.size(); e++)
		delete mesh.edges[e];
	for (int f = 0; f < mesh.faces.size(); f++)
		delete mesh.faces[f];
	mesh.verts.clear();
	mesh.nodes.clear();
	mesh.edges.clear();
	mesh.faces.clear();
}
