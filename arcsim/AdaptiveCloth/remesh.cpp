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

#include "remesh.hpp"
#include "blockvectors.hpp"
#include "geometry.hpp"
#include "io.hpp"
#include "magic.hpp"
#include "util.hpp"
#include <assert.h>
#include <cstdlib>
#include <cstdio>
using namespace std;

RemeshOp RemeshOp::inverse() const
{
	RemeshOp iop;
	iop.added_verts = removed_verts;
	iop.removed_verts = added_verts;
	iop.added_nodes = removed_nodes;
	iop.removed_nodes = added_nodes;
	iop.added_edges = removed_edges;
	iop.removed_edges = added_edges;
	iop.added_faces = removed_faces;
	iop.removed_faces = added_faces;
	return iop;
}

void RemeshOp::apply(Mesh &mesh) const
{
	// cout << "removing " << removed_faces << ", " << removed_edges << ", " << removed_verts << " and adding " << added_verts << ", " << added_edges << ", " << added_faces << endl;
	for (int i = 0; i < removed_faces.size(); i++)
		mesh.remove(removed_faces[i]);
	for (int i = 0; i < removed_edges.size(); i++)
		mesh.remove(removed_edges[i]);
	for (int i = 0; i < removed_nodes.size(); i++)
		mesh.remove(removed_nodes[i]);
	for (int i = 0; i < removed_verts.size(); i++)
		mesh.remove(removed_verts[i]);
	for (int i = 0; i < added_verts.size(); i++)
		mesh.add(added_verts[i]);
	for (int i = 0; i < added_nodes.size(); i++)
		mesh.add(added_nodes[i]);
	for (int i = 0; i < added_edges.size(); i++)
		mesh.add(added_edges[i]);
	for (int i = 0; i < added_faces.size(); i++)
		mesh.add(added_faces[i]);
}

void RemeshOp::done() const
{
	for (int i = 0; i < removed_verts.size(); i++)
		delete removed_verts[i];
	for (int i = 0; i < removed_nodes.size(); i++)
		delete removed_nodes[i];
	for (int i = 0; i < removed_edges.size(); i++)
		delete removed_edges[i];
	for (int i = 0; i < removed_faces.size(); i++)
		delete removed_faces[i];
}

ostream &operator<< (ostream &out, const RemeshOp &op)
{
	out << "removed " << op.removed_verts << ", " << op.removed_nodes << ", "
		<< op.removed_edges << ", " << op.removed_faces << ", added "
		<< op.added_verts << ", " << op.added_nodes << ", " << op.added_edges
		<< ", " << op.added_faces;
	return out;
}

template <typename T>
void compose_removal(T *t, vector<T*> &added, vector<T*> &removed)
{
	int i = find(t, added);
	if (i != -1)
	{
		remove(i, added);
		delete t;
	}
	else
		removed.push_back(t);
}

RemeshOp compose(const RemeshOp &op1, const RemeshOp &op2)
{
	RemeshOp op = op1;
	for (int i = 0; i < op2.removed_verts.size(); i++)
		compose_removal(op2.removed_verts[i], op.added_verts, op.removed_verts);
	for (int i = 0; i < op2.removed_nodes.size(); i++)
		compose_removal(op2.removed_nodes[i], op.added_nodes, op.removed_nodes);
	for (int i = 0; i < op2.removed_edges.size(); i++)
		compose_removal(op2.removed_edges[i], op.added_edges, op.removed_edges);
	for (int i = 0; i < op2.removed_faces.size(); i++)
		compose_removal(op2.removed_faces[i], op.added_faces, op.removed_faces);
	for (int i = 0; i < op2.added_verts.size(); i++)
		op.added_verts.push_back(op2.added_verts[i]);
	for (int i = 0; i < op2.added_nodes.size(); i++)
		op.added_nodes.push_back(op2.added_nodes[i]);
	for (int i = 0; i < op2.added_faces.size(); i++)
		op.added_faces.push_back(op2.added_faces[i]);
	for (int i = 0; i < op2.added_edges.size(); i++)
		op.added_edges.push_back(op2.added_edges[i]);
	return op;
}

// Fake physics for midpoint evaluation

Mat2x3 derivative_matrix(const Vec2 &u0, const Vec2 &u1, const Vec2 &u2)
{
	Mat2x2 Dm = Mat2x2(u1 - u0, u2 - u0);
	Mat2x2 invDm = Dm.inv();
	return invDm.t()*Mat2x3::rows(Vec3(-1, 1, 0), Vec3(-1, 0, 1));
}

double area(const Vec2 &u0, const Vec2 &u1, const Vec2 &u2)
{
	return wedge(u1 - u0, u2 - u0) / 2;
}

template <int n> struct Quadratic
{
	Mat<n * 3, n * 3> A;
	Vec<n * 3> b;
	Quadratic() : A(0), b(0) {}
	Quadratic(const Mat<n * 3, n * 3> &A, const Vec<n * 3> &b) : A(A), b(b) {}
};
template <int n>
Quadratic<n> &operator*= (Quadratic<n> &q, double a)
{
	q.A *= a; q.b *= a; return q;
}
template <int n>
Quadratic<n> &operator+= (Quadratic<n> &q, const Quadratic<n> &r)
{
	q.A += r.A; q.b += r.b; return q;
}
template <int n>
ostream &operator<< (ostream &out, const Quadratic<n> &q) { out << "<" << q.A << ", " << q.b << ">"; return out; }

template <int m, int n, int p, int q>
Mat<m*p, n*q> kronecker(const Mat<m, n> &A, const Mat<p, q> &B)
{
	Mat<m*p, n*q> C;
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			for (int k = 0; k < p; k++)
				for (int l = 0; l < q; l++)
					C(i*p + k, j*q + l) = A(i, j)*B(k, l);
	return C;
}

template <int m> Mat<m, 1> colmat(const Vec<m> &v)
{
	Mat<1, m> A; for (int i = 0; i < m; i++) A(i, 0) = v[i]; return A;
}
template <int n> Mat<1, n> rowmat(const Vec<n> &v)
{
	Mat<1, n> A; for (int i = 0; i < n; i++) A(0, i) = v[i]; return A;
}

template <Space s>
Quadratic<3> stretching(const Vert *vert0, const Vert *vert1,
						const Vert *vert2)
{
	const Vec2 &u0 = vert0->u, &u1 = vert1->u, &u2 = vert2->u;
	const Vec3 &x0 = pos<s>(vert0->node), &x1 = pos<s>(vert1->node),
		&x2 = pos<s>(vert2->node);
	Mat2x3 D = derivative_matrix(u0, u1, u2);
	Mat3x2 F = Mat3x3(x0, x1, x2)*D.t(); // = (D * Mat3x3(x0,x1,x2).t()).t()
	Mat2x2 G = (F.t()*F - Mat2x2(1)) / 2.;
	// eps = 1/2(F'F - I) = 1/2([x_u^2 & x_u x_v \\ x_u x_v & x_v^2] - I)
	// e = 1/2 k0 eps00^2 + k1 eps00 eps11 + 1/2 k2 eps11^2 + 1/2 k3 eps01^2
	// grad e = k0 eps00 grad eps00 + ...
	//        = k0 eps00 Du' x_u + ...
	Vec3 du = D.row(0), dv = D.row(1);
	Mat<3, 9> Du = kronecker(rowmat(du), Mat3x3(1)),
		Dv = kronecker(rowmat(dv), Mat3x3(1));
	const Vec3 &xu = F.col(0), &xv = F.col(1); // should equal Du*mat_to_vec(X)
	Vec<9> fuu = Du.t()*xu, fvv = Dv.t()*xv, fuv = (Du.t()*xv + Dv.t()*xu) / 2.;
	Vec<4> k;
	k[0] = 1;
	k[1] = 0;
	k[2] = 1;
	k[3] = 1;
	Vec<9> grad_e = k[0] * G(0, 0)*fuu + k[2] * G(1, 1)*fvv
		+ k[1] * (G(0, 0)*fvv + G(1, 1)*fuu) + k[3] * G(0, 1)*fuv;
	Mat<9, 9> hess_e = k[0] * (outer(fuu, fuu) + max(G(0, 0), 0.)*Du.t()*Du)
		+ k[2] * (outer(fvv, fvv) + max(G(1, 1), 0.)*Dv.t()*Dv)
		+ k[1] * (outer(fuu, fvv) + max(G(0, 0), 0.)*Dv.t()*Dv
				  + outer(fvv, fuu) + max(G(1, 1), 0.)*Du.t()*Du)
		+ k[3] * (outer(fuv, fuv));
	// ignoring k[3]*G(0,1)*(Du.t()*Dv+Dv.t()*Du)/2.) term
	// because may not be positive definite
	double a = area(u0, u1, u2);
	return Quadratic<3>(a*hess_e, a*grad_e);
}

double area(const Vec3 &x0, const Vec3 &x1, const Vec3 &x2)
{
	return norm(cross(x1 - x0, x2 - x0)) / 2;
}
Vec3 normal(const Vec3 &x0, const Vec3 &x1, const Vec3 &x2)
{
	return normalize(cross(x1 - x0, x2 - x0));
}
double dihedral_angle(const Vec3 &e, const Vec3 &n0, const Vec3 &n1)
{
	double cosine = dot(n0, n1), sine = dot(e, cross(n0, n1));
	return -atan2(sine, cosine);
}

template <Space s>
Quadratic<4> bending(double theta0, const Vert *vert0, const Vert *vert1,
					 const Vert *vert2, const Vert *vert3)
{
	const Vec3 &x0 = pos<s>(vert0->node), &x1 = pos<s>(vert1->node),
		&x2 = pos<s>(vert2->node), &x3 = pos<s>(vert3->node);
	Vec3 n0 = normal(x0, x1, x2), n1 = normal(x1, x0, x3);
	double theta = dihedral_angle(normalize(x1 - x0), n0, n1);
	double l = norm(x0 - x1);
	double a0 = area(x0, x1, x2), a1 = area(x1, x0, x3);
	double h0 = 2 * a0 / l, h1 = 2 * a1 / l;
	double w_f0v0 = dot(x2 - x1, x0 - x1) / sq(l),
		w_f0v1 = 1 - w_f0v0,
		w_f1v0 = dot(x3 - x1, x0 - x1) / sq(l),
		w_f1v1 = 1 - w_f1v0;
	Vec<12> dtheta = mat_to_vec(Mat<3, 4>(-(w_f0v0*n0 / h0 + w_f1v0 * n1 / h1),
										  -(w_f0v1*n0 / h0 + w_f1v1 * n1 / h1),
										  n0 / h0,
										  n1 / h1));
	double ke = 1;
	double shape = 1;//sq(l)/(2*(a0 + a1));
	return Quadratic<4>((a0 + a1) / 4 * ke*shape*outer(dtheta, dtheta) / 2.,
		(a0 + a1) / 4 * ke*shape*(theta - theta0)*dtheta / 2.);
}

template <int n> Quadratic<1> restrict(const Quadratic<n> &q, int k) {
	Quadratic<1> r;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
			r.A(i, j) = q.A(k * 3 + i, k * 3 + j);
		r.b[i] = q.b[k * 3 + i];
	}
	return r;
}

template <Space s>
void set_midpoint_position(const Edge *edge, Vert *vnew[2], Node *node)
{
	Quadratic<1> qs, qb;
	for (int i = 0; i < 2; i++)
	{
		if (!edge->adjf[i])
			continue;
		const Vert *v0 = edge_vert(edge, i, i),
			*v1 = edge_vert(edge, i, 1 - i),
			*v2 = edge_opp_vert(edge, i),
			*v = vnew[i];
		qs += restrict(stretching<s>(v0, v, v2), 1);
		qs += restrict(stretching<s>(v, v1, v2), 0);
		qb += restrict(bending<s>(0, v, v2, v0, v1), 0);
		// if (S == WS) REPORT(qb);
		const Edge *e;
		e = get_edge(v0->node, v2->node);
		if (const Vert *v4 = edge_opp_vert(e, e->nodes[0] == v0->node ? 0 : 1))
			qb += restrict(bending<s>(e->theta_ideal, v0, v2, v4, v), 3);
		// if (S == WS) REPORT(qb);
		e = get_edge(v1->node, v2->node);
		if (const Vert *v4 = edge_opp_vert(e, e->nodes[0] == v1->node ? 1 : 0))
			qb += restrict(bending<s>(e->theta_ideal, v1, v2, v, v4), 2);
		// if (S == WS) REPORT(qb);
	}
	if (edge->adjf[0] && edge->adjf[1])
	{
		const Vert *v2 = edge_opp_vert(edge, 0), *v3 = edge_opp_vert(edge, 1);
		double theta = edge->theta_ideal;
		qb += restrict(bending<s>(theta, edge_vert(edge, 0, 0), vnew[0],
								  v2, v3), 1);
		// if (S == WS) REPORT(qb);
		qb += restrict(bending<s>(theta, vnew[1], edge_vert(edge, 1, 1),
								  v2, v3), 0);
		// if (S == WS) REPORT(qb);
	}
	Quadratic<1> q;
	q += qs;
	q += qb;
	q.A += Mat3x3(1e-3);
	// if (S == WS) {
	//     REPORT(pos<S>(node));
	//     REPORT(qs.A);
	//     REPORT(qs.b);
	//     REPORT(qb.A);
	//     REPORT(qb.b);
	//     REPORT(-q.A.inv()*q.b);
	// }
	pos<s>(node) -= q.A.inv()*q.b;
}

// The actual operations

int combine_label(int l0, int l1) { return (l0 == l1) ? l0 : 0; }

RemeshOp split_edge(Edge* edge)
{
	RemeshOp op;
	Node *node0 = edge->nodes[0],
		*node1 = edge->nodes[1],
		*node = new Node((node0->y + node1->y) / 2.,
		(node0->x + node1->x) / 2.,
						 (node0->v + node1->v) / 2.,
						 combine_label(node0->label, node1->label));
	node->acceleration = (node0->acceleration + node1->acceleration) / 2.;
	op.added_nodes.push_back(node);
	op.removed_edges.push_back(edge);
	op.added_edges.push_back(new Edge(node0, node, edge->theta_ideal,
									  edge->label));
	op.added_edges.push_back(new Edge(node, node1, edge->theta_ideal,
									  edge->label));
	Vert *vnew[2] = { NULL, NULL };
	for (int s = 0; s < 2; s++)
	{
		if (!edge->adjf[s])
			continue;
		Vert *v0 = edge_vert(edge, s, s),
			*v1 = edge_vert(edge, s, 1 - s),
			*v2 = edge_opp_vert(edge, s);
		if (s == 0 || is_seam_or_boundary(edge))
		{
			vnew[s] = new Vert((v0->u + v1->u) / 2.,
							   combine_label(v0->label, v1->label));
			connect(vnew[s], node);
			op.added_verts.push_back(vnew[s]);
		}
		else
			vnew[s] = vnew[0];
		op.added_edges.push_back(new Edge(v2->node, node));
		Face *f = edge->adjf[s];
		op.removed_faces.push_back(f);
		op.added_faces.push_back(new Face(v0, vnew[s], v2, f->label));
		op.added_faces.push_back(new Face(vnew[s], v1, v2, f->label));
	}
	if (!::magic.preserve_creases)
	{
		set_midpoint_position<PS>(edge, vnew, node);
		set_midpoint_position<WS>(edge, vnew, node);
	}
	return op;
}

RemeshOp collapse_edge(Edge* edge, int i)
{
	RemeshOp op;
	Node *node0 = edge->nodes[i], *node1 = edge->nodes[1 - i];
	op.removed_nodes.push_back(node0);
	for (int e = 0; e < node0->adje.size(); e++)
	{
		Edge *edge1 = node0->adje[e];
		op.removed_edges.push_back(edge1);
		Node *node2 = (edge1->nodes[0] != node0) ? edge1->nodes[0] : edge1->nodes[1];
		if (node2 != node1 && !get_edge(node1, node2))
			op.added_edges.push_back(new Edge(node1, node2, edge1->theta_ideal,
											  edge1->label));
	}
	for (int s = 0; s < 2; s++)
	{
		Vert *vert0 = edge_vert(edge, s, i), *vert1 = edge_vert(edge, s, 1 - i);
		if (!vert0 || (s == 1 && vert0 == edge_vert(edge, 0, i)))
			continue;
		op.removed_verts.push_back(vert0);
		for (int f = 0; f < vert0->adjf.size(); f++)
		{
			Face *face = vert0->adjf[f];
			op.removed_faces.push_back(face);
			if (!is_in(vert1, face->v))
			{
				Vert *verts[3] = { face->v[0], face->v[1], face->v[2] };
				replace(vert0, vert1, verts);
				op.added_faces.push_back(new Face(verts[0], verts[1], verts[2],
												  face->label));
			}
		}
	}
	return op;
}

RemeshOp flip_edge(Edge* edge)
{
	RemeshOp op;
	Vert *vert0 = edge_vert(edge, 0, 0), *vert1 = edge_vert(edge, 1, 1),
		*vert2 = edge_opp_vert(edge, 0), *vert3 = edge_opp_vert(edge, 1);
	Face *face0 = edge->adjf[0], *face1 = edge->adjf[1];
	op.removed_edges.push_back(edge);
	op.added_edges.push_back(new Edge(vert2->node, vert3->node,
									  -edge->theta_ideal, edge->label));
	op.removed_faces.push_back(face0);
	op.removed_faces.push_back(face1);
	op.added_faces.push_back(new Face(vert0, vert3, vert2, face0->label));
	op.added_faces.push_back(new Face(vert1, vert2, vert3, face1->label));
	return op;
}
