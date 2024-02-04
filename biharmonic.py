from ngsolve import *
from ngsolve.webgui import Draw
mesh = Mesh (unit_square.GenerateMesh(maxh=0.25))

order = 4

V1 = H1(mesh, order=order, dirichlet="left|bottom|right|top")
V2 = NormalFacetFESpace(mesh, order=order-1, dirichlet="left|bottom|right|top")
V = V1*V2

#smooth solution data to test whether code is working
exu = x * y*(1-y)*(1-x)
exu_x = (2*x - 1)*(y - 1)*y
exu_y = (2*y - 1)*x*(x - 1);
force = 8;

w,what = V.TrialFunction()
v,vhat = V.TestFunction()

n = specialcf.normal(2)
h = specialcf.mesh_size

def jumpdn(v,vhat):
    return n*(grad(v)-vhat)
def hesse(v):
    return v.Operator("hesse")
def hessenn(v):
    return InnerProduct(n, hesse(v)*n)
    
def hessennBound(v):
    return InnerProduct(n, hesse(v.Trace())*n)

dS = dx(element_boundary=True)
a = BilinearForm(V)
a += InnerProduct (hesse(w), hesse(v)) * dx \
     - hessenn(w) * jumpdn(v,vhat) * dS \
     - hessenn(v) * jumpdn(w,what) * dS \
     + 3*order*order/h * jumpdn(w,what) * jumpdn(v,vhat) * dS
a.Assemble()

f = LinearForm(V)
f += force*v*dx
f += (-hessenn(v)+3*order*order/h*grad(v)*n)*(exu_x*n[0]+exu_y*n[1])*ds
#Note: ds integrates over the boundary of the domain (dS integrates over all element boundaries)
f.Assemble()

u = GridFunction(V)
u.vec.data = a.mat.Inverse(V.FreeDofs()) * f.vec

diff_u = u.components[0] - exu
err_u = sqrt(Integrate(InnerProduct(diff_u, diff_u), mesh))
print(err_u)

