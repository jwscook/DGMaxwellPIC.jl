# Discontinuous Galerkin 1D3V PIC



## DG Maxwell

Work out formulation for DG Maxwell first
$$
\frac{\partial E}{\partial t} - s^2 \nabla \times B = -s^2\mu_0 J
$$

$$
\frac{\partial B}{\partial t} + \nabla \times E = 0
$$

Now express fields $F$ such that $F(x)=\sum_{\hat e\in x,y,z}\sum_i \hat e c_i \phi_i(x)$, i.e. the different components are represented by a sum of basis functions scaled by a constant factor in each cell.

Use $\phi$ for trial function and $\varphi$ for test function (even though they're the same function)

The nth Lagrange basis function is zero at every Gauss Lobatto node except the nth, which makes it easy to integrate i.e. just the nth coefficient multiplied by the nth weight. This is why they're used very frequently.

Lets take the weak form of Ampere's equation
$$
\int \frac{\partial E}{\partial t}\varphi dV - c^2 \int (\nabla \times B) \varphi dV = -\int s^2\mu_0 J \varphi dV
$$
And then do Stoke's law where $\int_V \nabla \times A dV = \int_S \hat n \times A dS$ where $\hat n$ is the unit normal of the surface $dS$,
$$
\int \frac{\partial E}{\partial t}\varphi dV - c^2 \int (\nabla \cdot \hat n \times B) \varphi dS =- \int s^2\mu_0 J \varphi dV
$$
we get a flux $ \vec F = F_i \hat e_i\equiv - c^2 \hat n_i \times B$, 
$$
\int \frac{\partial E}{\partial t}\varphi dV - c^2 \int (\nabla \cdot \hat n \times B) \varphi dS = -\int \mu_0 J \varphi dV
$$
to
$$
\int \frac{\partial E}{\partial t}\varphi dV + \int (\nabla \cdot F) \varphi dS = -\int s^2 \mu_0 J \varphi dV
$$
then
$$
\int \frac{\partial E}{\partial t}\varphi dV - \int F \cdot \nabla \varphi dV + \int (\hat n \cdot F) \varphi dS =- \int s^2\mu_0 J \varphi dV
$$


because $\int \nabla\cdot (f \vec A) dV = \int (f \nabla\cdot \vec A + \vec A \cdot \nabla f) dV = \int f \vec A \cdot \vec n dS$.

Then turn the 3rd term on the LHS into a "numerical flux" (subtract an $F^*$) by doing an integration by parts swiftly followed by a reversal of the integration by parts to rustle up this $F^*$ chap, and 
$$
\int \frac{\partial E}{\partial t}\varphi dV + \int \varphi \nabla \cdot F dV - \int (\hat n \cdot (F-F^*) \varphi dS =- \int s^2 \mu_0 J \varphi dV.
$$
So we see that we have a time derivative term, a volumetric flux time (different from but also similar to the Continuous Galerkin term), a surface term (just like a finite volume term but where $\varphi$ isn't necessarily unity) and the volumetric current source term. Should $F-F^*=0$ then we're back to where we started, if not, then the term can be exploited to couple cells, which is the key to the success of the method.

we get a flux , Let's now consider just the first vector component of this latest equation.
$$
\int \varphi \frac{\partial E_1}{\partial t} dV + \int \varphi \nabla \cdot F dV - \int \varphi (\hat n \cdot (F-F^*) dS = -\int \varphi s^2\mu_0 J_1 dV.
$$

### Move from curl of a thing to a divergence of a flux of a something related,

that is $\nabla \times \vec A$ suddenly goes to $\nabla \cdot F$.

This deserves more of an explanation. It's all because DG wants to couple cells via a a flux through the shared faces through neighbouring cells. Hence we go from a curl to a divergence somehow. Textbooks often gloss over this and make various assumptions like Cartesian coordinates only. Let's take a closer look in the lingo of D'haeseleer et al
$$
\nabla \times \vec A = \frac{\epsilon_{ijk}}{\mathcal{J}}\frac{\partial A_j}{\partial u^i}\vec e_k
$$
How do we make that look like a divergence of a flux?
$$
\nabla \cdot \vec F = \frac{1}{\mathcal{J}}\frac{\partial \mathcal{J} F^i}{\partial u^i}
$$
First notice that $n\times A = \frac{1}{\mathcal{J}}\sum_i (n_j A_k - n_k A_j)\vec e_i \equiv G^i\vec e_i$ and take the divergence of this new $\vec G$ i.e. $\nabla \cdot \vec G = \frac{1}{\mathcal{J}}\frac{\partial \mathcal{J} G^i}{\partial u^i}$. Abusing notation just a smidgen we get
$$
\nabla \cdot \vec G = \frac{1}{\mathcal{J}}\frac{\partial }{\partial u^i}(n_j A_k - n_k A_j)^i
$$
where the sum is implicit. From this we can avenge the notation
$$
\vec G(i) \hat e_i = G^i e_i = \frac{1}{\mathcal{J}}(n_j A_k - n_k A_j)^i e_i \quad i,j,k\,\mathrm{cyc}\, 1,2,3
$$
It's clear that $F$ is a tensor, so let's figure out what it looks like (where $S$ is a current source term)
$$
\frac{\partial E}{\partial t} \cdot e^k = \frac{\partial E^k}{\partial t} = - (\nabla \cdot F)\cdot e^k + S
$$


so we must equate


$$
(\nabla \cdot F)\cdot e^k = - s^2 \nabla \times B \cdot e^k
$$

$$
(\nabla \cdot F)\cdot e^k = - s^2\frac{\epsilon_{ijk}}{\mathcal{J}}\frac{\partial B_j}{\partial u^i}\vec e_k \cdot e^k
$$

$$
(\nabla \cdot F)\cdot e^k = - s^2 \frac{\epsilon_{ijk}}{\mathcal{J}}(\frac{\partial B_j}{\partial u^i} - \frac{\partial B_i}{\partial u^j})
$$

Hence $F$ looks like
$$
-s^2\left((\hat e_1 \times B); (\hat e_2 \times B); (\hat e_3 \times B) \right)
$$
where $;$ indicate vertical stacking into a (column) vector, hence $\nabla \cdot F$ looks like
$$
\nabla \cdot F = -s^2\left(
\begin{array}{c}
\nabla \cdot (\hat e_1 \times B)\\
\nabla \cdot (\hat e_2 \times B)\\
\nabla \cdot (\hat e_3 \times B)
\end{array}
\right)
$$
which in D'haeseleer et al is, with $\hat e_i = n^i e_i$,


$$
\nabla \cdot F = -\frac{s^2}{\mathcal J}\left(\begin{array}{c}
\frac{\partial}{\partial u^i} (n^2 B^3 - n^3 B^2)^i\\
\frac{\partial}{\partial u^i} (n^3 B^1 - n^1 B^3)^i\\
\frac{\partial}{\partial u^i} (n^1 B^2 - n^2 B^1)^i\\
\end{array}\right)
$$
where the $i$ takes the value of the $e^j$; 
$$
\nabla \cdot F = -\frac{s^2}{\mathcal J}\left(\begin{array}{c}
\frac{\partial}{\partial u^2} (n^2 B^3) - \frac{\partial}{\partial u^3}(n^3 B^2)\\
\frac{\partial}{\partial u^3} (n^3 B^1) - \frac{\partial}{\partial u^1}(n^1 B^3)\\
\frac{\partial}{\partial u^1} (n^1 B^2) - \frac{\partial}{\partial u^2}(n^2 B^1)\\
\end{array}\right)
$$
And in Cartesian  coordinates or other orthogonal unit-Jacobian coordinates systems:


$$
\nabla \cdot F = -s^2\left(\begin{array}{c}
\frac{\partial B^z}{\partial y} - \frac{\partial B^y}{\partial u^z}\\
\frac{\partial B^x}{\partial z} - \frac{\partial B^z}{\partial u^x}\\
\frac{\partial B^y}{\partial x} - \frac{\partial B^x}{\partial u^y}\\
\end{array}\right)
$$
Which is the familiar and expected expression for $-c^2 \nabla \times B$. 

What about general curvilinear coordinate systems? To retrieve the curl we need $n^j$ to disappear, which means both $n^j = 1$ and $\frac{\partial  n^j}{\partial u^j} =0$, which makes life Cartesian (perhaps not strictly Cartesian - there may be other options but that doesn't help us in general). How is it generalised?

We would want to solve for the divergences of a tuple of vectors (note I'm not using the word tensor here because I just want a list of vectors). Let me explain (using $\hat E_u$ as notation for the vector component in the $u$ direction out of components $u,v,w$ associated with the unit vector $\hat u$ i.e. $\hat E_u$ has units of what ever you expect $E$ to have)
$$
\frac{\partial}{\partial t}\left(\begin{array}{c}
\hat E_u \hat u \\
\hat E_v \hat v \\
\hat E_w \hat w \\
\end{array}\right) 
= s^2 \nabla \times B|_{u,v,w} + s^2\mu_0 J|_{u,v,w}
= \frac{s^2}{\mathcal{J}}\epsilon_{ijk}\frac{\partial}{\partial u^i} B_j e_k + s^2\mu_0 J|_{u,v,w}
$$
where $|_{u,v,w}$ denotes the decomposition into $u,v,w$ components. Compare against the expression for divergence (and ignoring the current source term because it keeps getting in the way)
$$
\frac{\partial}{\partial t}\hat E_u \hat u = 
\frac{s^2}{\mathcal{J}}\epsilon_{iju}\frac{\partial}{\partial u^i} B_j e_u = 
\frac{s^2}{\mathcal{J}}\left(\frac{\partial}{\partial u^v} B_w - \frac{\partial}{\partial u^w} B_v\right) e_u
$$
hence we find that it is convenient to work with
$$
\mathcal{J}\frac{\partial}{\partial t}\hat E_u \hat u = 
 s^2\left(\frac{\partial}{\partial u^v} B_w - \frac{\partial}{\partial u^w} B_v\right) e_u =
 s^2\left(\frac{\partial}{\partial u^i} {\mathcal J} U^i\right) e_u
$$
where $U^i e_i$ is the $u$ component of the tuple of flux vectors. Hence:
$$
{\mathcal J} U^u = 0\\
{\mathcal J} U^v = B_w\\
{\mathcal J} U^w = -B_v\\
$$
And could write down similar expressions for the $V$ and $W$ components:
$$
{\mathcal J}F^i = 
{\mathcal J}\left(\begin{array}{c}
U^i& V^i& W^i
\end{array}\right)  = 
{\mathcal J}\left(\begin{array}{c}
U^u& V^u& W^u\\
U^v& V^v& W^v \\
U^w& V^w& W^w \\
\end{array}\right)
$$
so
$$
\mathcal{J}U^i = \epsilon_{ij1}B_j\\
\mathcal{J}V^i = \epsilon_{ij2}B_j\\
\mathcal{J}W^i = \epsilon_{ij3}B_j
$$


where $\epsilon_{lmn}$ is the Levi-Civita symbol.

See also [Kopriva et al](https://arxiv.org/pdf/1809.05206.pdf) for a nice demonstration of solving a conservation equation, in their case arbitrary ones e.g. Euler / Navier-Stokes, for an explanation on how this is done for DGSEM in combination with high order elements.
$$
\mathcal{J}\frac{\partial}{\partial t}E^k -
 s^2 \epsilon_{ijk}\frac{\partial}{\partial u^i} {\mathcal J} B_j = \mathcal{J}s^2\mu_0 J^k\\
 \mathcal{J}\frac{\partial}{\partial t}B^k + \epsilon_{ijk}\frac{\partial}{\partial u^i} {\mathcal J} E_j=0\\
$$
or in the lingo of a divergence of flux
$$
\mathcal{J}\frac{\partial}{\partial t}\left(\begin{array}{c}
E^k\\ B^k
\end{array}\right) + u^k \nabla_k \cdot 
\left(\begin{array}{c}
F \\
\end{array}\right)
  =
  \left(\begin{array}{c}\mathcal{J}s^2\mu_0 J^k\\0\end{array}\right)
$$
Where explicitly
$$
\mathcal{J}\frac{\partial}{\partial t}\left(\begin{array}{c}
E^u\\E^v\\E^w\\B^u\\B^v\\B^w
\end{array}\right) + 
\nabla \cdot \left(\begin{array}{c}
-s^2\epsilon_{iju}\frac{B_j}{\mathcal{J}}e_i \\
-s^2\epsilon_{ijv}\frac{B_j}{\mathcal{J}}e_i \\
-s^2\epsilon_{ijw}\frac{B_j}{\mathcal{J}}e_i \\
\epsilon_{iju}\frac{E_j}{\mathcal{J}}e_i \\
\epsilon_{ijv}\frac{E_j}{\mathcal{J}}e_i \\
\epsilon_{ijw}\frac{E_j}{\mathcal{J}}e_i \\
\end{array}\right)  =  \left(\begin{array}{c}
\mathcal{J}s^2\mu_0 J^u\\
\mathcal{J}s^2\mu_0 J^v\\
\mathcal{J}s^2\mu_0 J^w\\
0\\0\\0\\\end{array}\right)
$$
carrying on ad nauseam
$$
\mathcal{J}\frac{\partial}{\partial t}\left(\begin{array}{c}
E^u\\E^v\\E^w\\B^u\\B^v\\B^w
\end{array}\right) + 
\nabla \cdot \left(\begin{array}{c}
-\frac{s^2}{\mathcal{J}}\left(0 e_u + B_w e_v - B_v e_w\right)\\
-\frac{s^2}{\mathcal{J}}\left(-B_we_u+0e_v+B_ue_w\right)\\
-\frac{s^2}{\mathcal{J}}\left(B_ve_u -e_v B_u + 0e_w\right)\\
\left(0 e_u + E_w e_v - E_v e_w\right)\\
\left(-E_we_u+0e_v+E_ue_w\right)\\
\left(E_ve_u -e_v E_u + 0e_w\right)\\
\end{array}\right)  =  \left(\begin{array}{c}
\mathcal{J}s^2\mu_0 J^u\\
\mathcal{J}s^2\mu_0 J^v\\
\mathcal{J}s^2\mu_0 J^w\\
0\\0\\0\\\end{array}\right)
$$

## A Cartesian 2D 3V PIC code

Let's implement using Cartesian coordinates having grappled with the general form. 1D is a bit too simple, 3D is a bit long winded, so settle for 2D.

We need to solve for 3 components of the electric field and 3 components of the magnetic field, which all vary in 2D.
$$
\vec{E} = {E_i\vec{u}^i}\\
\vec{B} = {B_i\vec{u}^i}
$$
Going back to before I got distracted
$$
\int \varphi \frac{\partial E_i}{\partial t} dV + \int \varphi \nabla \cdot F_i dV - \int \varphi (\hat n \cdot (F-F^*)_i) dS = -\int \varphi s^2\mu_0 J_i dV.
$$

Let's figure out what this means for $E_x=\sum_j c_j l_j(x)$  where I'm using $j$ as an index to prepare for the fact that $c_j$ will be unknowns in a column vector pre-multiplied by a matrix whose columns correspond to trial functions (the unknowns) and rows to the test functions. Implying the integrals and inferring that $<l_i l_j>_{\tau}$ and similar really means $\int l_i l_j d\tau$ equals a matrix of integrals for row $i$ and column $j$. 
$$
<l_il_j \frac{\partial }{\partial t}>_V c_j - c^2\left(<l^B_i \frac{\partial l_{j}}{\partial y}>_V c^B_z - <l^B_i \frac{\partial l_{j}}{\partial z}>_Vc^B_y \right) +  <l_i\mathcal{F}_{j}>_{S_x} c_j + <l_i\mathcal{F}_{j}>_{S_y} c_j = -s^2\mu_0 <l_i l_j>_S J_j.
$$
where the flux $\mathcal{F}$ implies numerical fluxes we get a flux , through the surface. All the volume integrals are $i$ by $j$ matrices that can be scaled up and down according to cell size. The surface integrals, representing flux terms, are linear too and they couple cells together. A flux term looks like this:
$$
\left(\frac{1}{\bar Z}\alpha\left[\Delta \vec E - \hat n (\hat n \cdot {\Delta {\vec E}})\right]+ Z^{\\+} \hat n \times \Delta \vec H \right)\\
\left(\frac{1}{\bar Y}\alpha\left[\Delta \vec H - \hat n (\hat n \cdot {\Delta {\vec H}})\right]- Y^{\\+} \hat n \times \Delta \vec E \right)
$$
where $Z^\pm=\sqrt \frac{\mu^\pm}{\epsilon^\pm}$ and $Y^\pm = 1/Z^\pm$ and $\bar Q=Q^{\\+} +Q^{\\-}$ where $\pm$ indicate the different sides of the interface which have different permeabilities and permitivities in materials to which this formalism is usually applied, and $B=\mu H$. $\hat n$ is the unit vector of the surfaces surrounding each cell point from $-$ to $+$. Changing the above equation in a superficial manner to appear closer to the rest i.e. $B$ not $H$ and we don't consider relative permitivities/permeabilities because that's what the particles are for. Noting that $ZH=\sqrt{\frac{\mu_0}{\epsilon_0}}\mu_0 B=\frac{1}{\sqrt{\epsilon_0 \mu_0}} B = s B$, and also changing $c$ to $s$ to stand for the speed of light.
$$
\left(\frac{1}{2}s\epsilon_0\alpha\left[\Delta \vec E - \hat n (\hat n \cdot {\Delta {\vec E}})\right]+ s\hat n \times \Delta \vec B \right)\\
\left(\frac{1}{2}s\alpha\left[\Delta \vec B - \hat n (\hat n \cdot {\Delta {\vec B}})\right]- s \epsilon_0 \hat n \times \Delta \vec E \right)
$$
where $0 \leq \alpha \leq 1$ controls the upwindedness (0 for conservative central fluxes, 1 for upstream). Let's apply that to the $E_i$ part (so top equation for flux)
$$
\frac{1}{2}s\epsilon_0\alpha\left[\sum_{i'}(E_{i'}^{\\+}-E_{i'}^{\\-}) - \hat n_i \sum_{j} \hat n_i \cdot({E_j^{\\+}-E_j^{\\-}})\right]+ s \epsilon_{ijk}([B_k^{\\+}-B_k^{\\-}]-[B_j^{\\+}-B_j^{\\-}])
$$
assuming cyclic $i,j,k$ in the last term. So...
$$
\begin{array}{c}
\mathcal{F}^{E_x}_{j}|_{S_x} =& 0\\
\mathcal{F}^{E_x}_{j}|_{S_y} =& \frac{1}{2}s\epsilon_0\alpha\left[(E_x^{\\+}-E_x^{\\-}) - ({E_{y}^{\\+}-E_{y}^{\\-}})\right] + s (B_z^{\\+}-B_z^{\\-})\\
\mathcal{F}^{E_x}_{j}|_{S_z} =& \frac{1}{2}s\epsilon_0\alpha\left[(E_x^{\\+}-E_x^{\\-}) - ({E_{z}^{\\+}-E_{z}^{\\-}})\right] - s (B_y^{\\+}-B_y^{\\-})\\
\end{array}
$$
which simplifies to 
$$
F_j^{E_u}|_{S}=\sum_{\hat n} F_j^{E_u}|_{S_n}=\sum_{\hat n \in \mathrm{faces}}\frac{1}{2}s\epsilon_0\alpha\left[(E_u^{\\+}-E_u^{\\-}) - ({E_{n}^{\\+}-E_{n}^{\\-}})\right] + s \hat u \cdot \hat n \times \Delta \vec B
$$

$$
<l_i F_j^{E_u}>_{S}=\sum_{\hat n} <l_iF_j^{E_u}>_{S_n}=\sum_{\hat n \in \mathrm{faces}}<l_i\frac{1}{2}s\epsilon_0\alpha\left[(l_jc_{E_u}^{\\+}-l_jc_{E_u}^{\\-}) - ({l_j c_{E_n}^{\\+}-l_j c_{E_n}^{\\-}})\right] + s \hat u \cdot \hat n \times \Delta l_j c_{\vec B}>
$$

and the flux for Faraday's equation
$$
<l_i F_j^{B_u}>_{S}=\sum_{\hat n \in \mathrm{faces}}<l_i\frac{1}{2}s\alpha\left[(l_jc_{B_u}^{\\+}-l_jc_{B_u}^{\\-}) - ({l_j c_{B_n}^{\\+}-l_j c_{B_n}^{\\-}})\right] - s\epsilon_0\hat u \cdot \hat n \times \Delta l_j c_{\vec E}
$$
Each cell contains N nodes with each one associated with a Lagrange polynomial. The nodes along each face are associated with a non-zero polynomial and it's these that contribute to the flux. These are the dofs that couple cells. The ghost cells for each node must come from neighbours or the boundary condition via the boundary manager.

In 1D only the first and last dof per cell is non-zero at the boundary. In 2D only the dofs corresponding to the edge nodes are non-zero at the boundary.

The fluxes are the most difficult bit. Let $s_i$ still represent the speed of light, but the subscript indicates it's associated with a flux in direction $i$. $w_i$ will be the cubature weight for the $i^{th}$ face / direction. Here $a_i=\frac{1}{2}s\alpha\epsilon_0$ from the $i^{th}$ direction flux. Similarly  $b_i=\frac{1}{2}s\alpha$. The matrix below shows which numbers go in where to connect fluxes between cells.
$$
\frac{\partial}{\partial t}
\left[\begin{array}{c}
E_x\\
E_y\\
E_z\\
B_x\\
B_y\\
B_z\\
\end{array}\right]
\propto
\left[\begin{array}{c} 
a_y + a_z & -a_y & -a_z & 0 & -s_z & s_y \\
-a_x & a_x + a_z & -a_z & s_z & 0 & s_x \\
-a_x & -a_y & a_x + a_y & -s_y & - s_x & 0 \\
0 & s_z \epsilon_0 & - s_y \epsilon_0 & b_y + b_z & -b_y & -b_z \\
- s_z \epsilon_0 & 0 & - s_x \epsilon_0 & -b_x & b_x + b_z & -b_z \\
s_y \epsilon_0 & s_x \epsilon_0 & 0 & -b_x & -b_y & b_x + b_y \\
\end{array}\right]
\left[\begin{array}{c}
\Delta E_x\\
\Delta E_y\\
\Delta E_z\\
\Delta B_x\\
\Delta B_y\\
\Delta B_z\\
\end{array}\right]
$$
What about the $\int \varphi \nabla \cdot F_i dV$ term?

### Simple single variable 1D example

$$
\frac{\partial f}{\partial t} + v\frac{\partial f}{\partial x}=0
$$

where the value of $f$ in cell $c$ is the sum of lagrange functions $f_c=\sum_j l_j$. Let's say there are third order Lagrange functions in each cell. The flux is just $v$ through each face. Choosing a non-boundary cell


$$
\left[\begin{array}{c}
&&...\\
...F^+ & 0 & 0 & 0 & 0\\
-F^- & F^+ & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & -F^- & F^+\\
0 & 0 & 0 & 0 & -F^-... \\
&&...\\
\end{array}\right]
\left[\begin{array}{c}
...\\
\partial_x f_{c-1}^3\\
\partial_x f_{c}^1\\
\partial_x f_{c}^2\\
\partial_x f_{c}^3\\
\partial_x f_{c+1}^1\\
...
\end{array}\right]
$$



# Current source term

# Lagrange polynomials

The received wisdom is to use Lagrange polynomials because they play nicely with Gauss Lobatto quadrature;

- they are orthogonal polynomials with respect to a unit kernel
- the $n^{th}$ polynomial is zero at all nodes except at the $n^{th}$ quadrature point
- which means only 2 polynomials are ever non-zero at the boundaries (so only one polynomial couples)
- they have a nice recurrence relation for a generating function

These points mean that one can play lots of tricks to perform integrals and derivatives very efficiently.
$$
l_i(x) = c_i\Pi_{j\neq i}^n\frac{x-x_j}{x_i - x_j}
$$
where $l_j$ is the $j^{th}$  Lagrange polynomial of order $n$ and $x_i$ is the $i^{th}$ location of the where $(n-1)$ interpolating data are known  (which will be Gauss-Lobatto quadrature points for us).

Using an alternative form, which is handy because it offers us a formulation for the derivative, can be uncovered by noticing
$$
\frac{\partial }{\partial x}|_{x=x_i}\Pi_{j=1}^n(x-x_j)=\Pi_{j\neq i}^n(x_i-x_j)\equiv \lambda_i'(x_i)
$$
which defines $\lambda'$ which looks like the denominator of $l_j$ except it's missing a factor of $(x-x_j)$. Hence
$$
l_i(x) = c_i\Pi_{j\neq i}^n\frac{x-x_j}{x_i - x_j}=c_i \frac{\lambda(x)}{(x-x_i)\lambda'(x_i)}\equiv c_i\lambda_i(x)
$$
Now the full polynomial $L$ built up of the $n$-long list of  $l_i$ terms is
$$
L(x) = \sum_i c_i\lambda_i(x)=\sum_{j} c_i \frac{\lambda(x)}{(x-x_i)\lambda'(x_i)}
$$
The derivative of $L(x)$ can be calculated easily:
$$
\frac{\partial L(x)}{\partial x} = \sum_i [c_i'\lambda_i(x)+ c_i\lambda_i'(x)]\simeq \sum_i c_i\lambda_i'(x)
$$
but after some boring manipulation
$$
l_i'(x) = c_i\Pi_{j\neq i}^n\left(\frac{x-x_j}{x_i - x_j}\right)'=c_i l_i(x)\sum_{j\neq i}\frac{1}{x-x_j}
$$
Hence
$$
L'(x)=\sum_i l_i'(x) = \sum_i c_i l_i(x)\sum_{j\neq i}\frac{1}{x-x_j}
$$
which doesn't work at $x=x_i$. An alternative is
$$
L'(x)=\sum_i l_i'(x) = \sum_i c_i \sum_{j\neq i}\frac{1}{x_i-x_j}\prod_{k\neq i,j}\frac{x-x_k}{x_i-x_k}
$$

and the second derivative
$$
l_i''(x) = \sum_{j\neq i}\frac{1}{x_i-x_j}\sum_{k\neq i,j}\frac{1}{x_i-x_k}\prod_{l\neq i,j,k}\frac{x-x_l}{x_i-x_l}
$$
and so on. All derivatives can be precomputed as a dot product between coefficients $c$ and a vector of values that when dotted with $c$ gives the nth derivative at location $x$. 

The integrals of the Lagrange polynomials are easily calculated with Gauss-Legendre quadrature nodes and weights
$$
\int_{-1}^1 L(x)dx = \sum_i w_i\sum_j l_j(x_i) = \sum_j w_j c_j
$$

but they are not orthogonal with other types of nodes e.g. Lobatto.

## 