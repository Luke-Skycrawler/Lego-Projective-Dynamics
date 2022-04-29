import taichi as ti
import numpy as np 
# (x|y|r)\[(.)\]    bricks[$2].$1

dim = 2
max_bricks = 1000
chunk = 1
res = 512
grid = 16
radius = 20
Radius = 18
diameter = 2 * radius
zero = -1e-9
Diameter = diameter / res
worldl = lambda l: (l-1) * Diameter
n_jacobian_iters = 100
n2_jacobian_iters = 1
dt = 5e-4
debug = False
allow_cross = False
flight = 5e-3
ti.init(arch=ti.x64)
ex = ti.Vector([1.,0.])
ey = ti.Vector([0.,1.])
linear_dict = {
    'x': ti.Vector.field(dim, dtype = float, needs_grad = True), # end
    'y': ti.Vector.field(dim, dtype = float, needs_grad = True), # end
    'l': ti.field(dtype = ti.i32, needs_grad= True),  # length
}
contacts = ti.field(dtype = ti.i32)
p_x = ti.Vector.field(dim, dtype = float)
p_y = ti.Vector.field(dim, dtype = float)
p_vx = ti.Vector.field(dim, dtype = float)
p_vy = ti.Vector.field(dim, dtype = float)
s_x = ti.Vector.field(dim, dtype = float)
s_y = ti.Vector.field(dim, dtype = float)
v_x = ti.Vector.field(dim, dtype = float)
v_y = ti.Vector.field(dim, dtype = float)
q_x = ti.Vector.field(dim, dtype = float)
q_y = ti.Vector.field(dim, dtype = float)
q_vx = ti.Vector.field(dim, dtype = float)
q_vy = ti.Vector.field(dim, dtype = float)
bricks = ti.StructField(linear_dict)
container = ti.root.pointer(ti.i, max_bricks)
container.place(bricks)
container.place(p_x, p_y,p_vx, p_vy, s_x, s_y, v_x, v_y, q_x, q_y, q_vx, q_vy, contacts)
probes = ti.Vector.field(2, float, (grid,grid))
sdf = ti.field(float, (grid, grid))
tot_sdf = ti.field(float, ())
tmp_dense_matrix = ti.field(ti.u8, (100,100))


ti.root.lazy_grad()


@ti.func
def inv(c):    # 1/ c    
    return ti.Vector([c[1], -c[0]])

@ti.func
def R(r):
    return ti.Matrix.cols([r,inv(r)])
@ti.func
def signed_distance(x_j, I):  # I: index of brick; x_j: pos
    l,x,y = bricks[I].l, bricks[I].x, bricks[I].y,
    Rm = R((y-x).normalized())
    p = x_j - x
    p = Rm @ p
    ret = 0.
    if 0 < p.x < worldl(l):
        ret = ti.abs(p.y)
    else : 
        ret = ti.min(p.norm(), (p - worldl(l) * ex).norm())
    return ret

@ti.kernel
def init() -> ti.i32:
    for I in ti.grouped(probes):
        probes[I] = (I + 0.5) / grid
    x = ti.Vector([0.2, 0.05])
    y = x + ti.Vector([1., 0.]) * worldl(7)
    # x2 = ti.Vector([0.2 + worldl(7)/2, 0.2])
    x2 = ti.Vector([0.1, 0.5])
    # y2 = x2 + ti.Vector([1., 1.]).normalized() * worldl(7)
    y2 = x2 + ey * worldl(7)
    x3 = ti.Vector([0.5, (y2 + x2).y/2.])
    # y3 = x3 + ti.Vector([-1., 1.]).normalized() * worldl(7)
    y3 = x3 + ti.Vector([1., -0.5]).normalized() * worldl(7)
    # bricks[0] = ti.Struct({
    #     'x': x, 
    #     'y': y,
    #     'l': 7,  # length
    # })
    bricks[0] = ti.Struct({
        'x': x3, 
        'y': y3,
        'l': 7,  # length
    })
    bricks[1] = ti.Struct({
        'x': x2, 
        'y': y2,
        'l': 7,  # length
    })
    v_x[0] = v_y[0] = -ex * 5
    v_x[1] = v_y[1] = ex * 5

    return 2

@ti.kernel
def grid_sdf(t: ti.i32):
    for I in ti.grouped(sdf):
        sdf[I] = signed_distance(probes[I], t)
        tot_sdf[None] += sdf[I]
@ti.kernel
def test_wrapper(cnt: ti.i32):
    for i in ti.grouped(probes):
        probes.grad[i] = nebla_sdf(probes[i], cnt)


# def draw_all(ggui):
#     to_np = lambda x: bricks.get_member_field(x).to_numpy()
#     l,x,r = [to_np(i) for i in ['l','x','r']]
#     ggui.lines(r,l,x, radius) 
    
@ti.kernel
def preview_brick(arr:ti.ext_arr(), cnt: ti.i32):
    r = ti.Vector([arr[0],arr[1]]).normalized()
    bricks[cnt].y = r * worldl(bricks[cnt].l) + bricks[cnt].x

@ti.func
def update_fenwick(boolean, i, j): # collision at point x/y
    # pass
    tmp_dense_matrix[i,j] = boolean
    tmp_dense_matrix[j,i] = boolean

@ti.func
def contact(i,j):
    xc = signed_distance(bricks[i].x, j) < Diameter
    yc = signed_distance(bricks[i].y, j) < Diameter
    return xc or yc or contact_ev(i,j)

@ti.func
def contact_ev(i,j):
    return min(signed_distance(bricks[j].x, i), signed_distance(bricks[j].y, i)) < Diameter
    # return tmp_dense_matrix[i,j] & 2

@ti.kernel
def ve(i: ti.i32):
    for J in bricks:
        xc = signed_distance(bricks[i].x, J) < Diameter
        yc = signed_distance(bricks[i].y, J) < Diameter
        update_fenwick(xc or yc, i,J)

@ti.kernel
def ev(I:ti.i32):
    for j in bricks:
        t = min(signed_distance(bricks[j].x, I), signed_distance(bricks[j].y, I)) < Diameter
        # TODO: watch out the sdf 
        update_fenwick(t, j,I)
    
# @ti.kernel
# def _ee(i: ti.i32):
#     for j in bricks:
#         t = (bricks[j].r.transpose() @ bricks.x.grad[j]) * (bricks[j].r.transpose() @ bricks.y.grad[j]) < 0
#         update_fenwick(t[0], i,j)

# def sdf_gradient(I): # sdf gradient of brick I at ends of other bricks 
#     tot_sdf[None] = 0.
#     with ti.Tape(tot_sdf):
#         _grad_sdf(I)

# @ti.kernel
# def _grad_sdf(I:ti.i32):
#     for j in bricks:
#         tot_sdf[None] += signed_distance(bricks[j].x, I) + signed_distance(bricks[j].y, I)

@ti.kernel
def ee(i:ti.i32):
    for j in bricks:
        r1 = bricks[i].y - bricks[i].x
        r2 = bricks[j].y - bricks[j].x
        m11 = ti.Matrix.cols([bricks[j].x-bricks[i].x, r1])
        m12 = ti.Matrix.cols([bricks[j].y-bricks[i].y, r1])
        m21 = ti.Matrix.cols([bricks[j].x-bricks[i].x, r2])
        m22 = ti.Matrix.cols([bricks[j].y-bricks[i].y, r2])
        t = m11.determinant() * m12.determinant() < 0 and m21.determinant() * m22.determinant() < 0
        update_fenwick(t, i, j)

@ti.func
def nebla_sdf(x, J):
    r = (bricks[J].y - bricks[J].x).normalized()
    t =  R(r) @ (x - bricks[J].x)
    ret = ti.Vector.zero(float, 2)
    if 0 < t.x < worldl(7):
        ret = inv(r) * (1 if t.y > 0 else -1)
    elif t.x < 0:
        ret = (x - bricks[J].x).normalized()
    else:
        ret = (x - bricks[J].y).normalized()
    return ret
    
@ti.kernel
def copy_x_to_s(f_x: ti.template(), f_y: ti.template()):
    for i in bricks:
        f_x[i] = bricks[i].x
        f_y[i] = bricks[i].y

@ti.func
def contact_point_x_or_y(I,j):  # return 1 for x 0 for y
    boolean = signed_distance(bricks[j].x, I) < signed_distance(bricks[j].y, I)
    contact_point = bricks[j].x if boolean else bricks[j].y
    return contact_point, boolean

debug_solver = True
@ti.func
def solve_jacobian(A, b, x0): # solve x for A x = b
    T = A - ti.Matrix.identity(float, 5)
    for iter in range(n2_jacobian_iters):
        x1 = b - T @ x0
        if ti.static(debug_solver):
            print(f'iter {iter}, x1 - x0 = {(x1-x0).norm()}')
        x0 = x1
    return x0

# @ti.func
# def collision_matrix(lam, coord):
#     a21 = (lam - 0.5) / 2 - 1/12
#     a22 = (lam - 0.5) / 2 + 1/12

#     x, y = coord.x, coord.y
#     a43, a44, a45 = y, y, x
#     a54, a55 = x, -y
#     A = ti.Matrix.rows(
#         [[1., 1., 1., 1., 0.],
#          [a21, a22, 0., 0., 0.],
#          [lam, 1-lam, -1, 0., 0.],
#          [0., 0., a43, a44, a45],
#          [0., 0., 0., a54, a55]
#         ])
#     return A

@ti.func
def collision_matrix(lam):
    A = ti.Matrix.rows([
        [1., 1., 1., 1.],
        [lam - 1/3, lam - 2/3, 0., 0.],
        [lam, 1-lam, -1., 0.],
        [0., 0., 1., -1.]
    ])
    return A
@ti.kernel
def project_local(cnt: ti.i32):   # local step for all bricks 

    for I in bricks:
        p_x[I] = p_y[I] = ti.Vector.zero(float, 2)
        contacts[I] = 1
        p_x[I] += (bricks[I].y - bricks[I].x) * (1-worldl(7)/(bricks[I].y-bricks[I].x).norm()) / 2
        p_y[I] -= (bricks[I].y - bricks[I].x) * (1-worldl(7)/(bricks[I].y-bricks[I].x).norm()) / 2
        # boundary conditions
        p_x[I].y += max(Diameter /2 - bricks[I].x.y, 0)
        p_y[I].y += max(Diameter /2 - bricks[I].y.y, 0)
        p_x[I].x += max(Diameter /2 - bricks[I].x.x, 0)
        p_y[I].x += max(Diameter /2 - bricks[I].y.x, 0)
        p_x[I].x -= max(Diameter /2 - (1- bricks[I].x.x), 0)
        p_y[I].x -= max(Diameter /2 - (1- bricks[I].y.x), 0)

        for j in range(cnt):
            if j != I and contact(I,j):
                if contact_ev(I,j):
                    contact_point, _ = contact_point_x_or_y(I,j)
                    p_x[I] -= (Diameter - signed_distance(contact_point, I)) * nebla_sdf(contact_point, I).normalized()
                    p_y[I] -= (Diameter - signed_distance(contact_point, I)) * nebla_sdf(contact_point, I).normalized()
                else:   # contact ve
                    p_x[I] += max(-signed_distance(bricks[I].x, j) + Diameter, 0.) * nebla_sdf(bricks[I].x, j).normalized()
                    p_y[I] += max(-signed_distance(bricks[I].y, j) + Diameter, 0.) * nebla_sdf(bricks[I].y, j).normalized()
                contacts[I] += 1

@ti.func
def contact_point_on_edge2(i, contact_point):
    r = (bricks[i].y - bricks[i].x).normalized()
    t = R(r) @ (contact_point - bricks[i].x)
    lam = 1 - t[0]/worldl(7)
    edge_point = lam * bricks[i].y + (1-lam) * bricks[i].x
    edge_point += (1 if t[1] > 0 else -1) * Diameter/2 * inv(r)
    return edge_point, lam

@ti.func
def contact_point_on_edge(i, contact_point):                
    # contact point =  lam * x + (1 - lam) * y
    lam = 1. - (contact_point - bricks[i].x).dot((bricks[i].y - bricks[i].x).normalized())/worldl(7)
    return max(min(lam , 1.), 0.)
    
@ti.func
def perpendicular(rm, x):
    return (rm @ x).y

@ti.func
def tangent(rm, x):
    return (rm @ x).x

@ti.func
def cross_2d(n, r):
    return ti.abs(n.cross(r) * r.norm())

@ti.func
def boundary_down(sin_theta, n1, x, q_v, center):
    impact = 0.
    if Diameter/2 > x.y and q_v.y < 0: 
        ra = ti.Vector([x.x, 0.0]) - center
        impact = ti.abs(1 * q_v.y / (1 + 12 / worldl(7) ** 2 * cross_2d(ey, ra)))
        # p_vx[i] += impact * (ey + sin_theta * 6 * n1)
        # p_vy[i] += impact * (ey - sin_theta * 6 * n1)
        # print('boundary',impact)
    return impact * (ey + sin_theta * 6 * n1), impact * (ey - sin_theta * 6 * n1)

@ti.kernel
def project_v(cnt: ti.i32):
    for i in bricks:
        r1 = (bricks[i].y - bricks[i].x).normalized()
        px0 = q_vx[i] + (q_vy[i] - q_vx[i]).dot(r1) * r1 /2
        py0 = q_vy[i] - (q_vy[i] - q_vx[i]).dot(r1) * r1 /2
        p_vx[i], p_vy[i] = px0, py0

        # p_vx[j] = q_vx[j] + (q_vy[j] - q_vx[j]).dot(r2) * r2 /2
        # p_vy[j] = q_vy[j] - (q_vy[j] - q_vx[j]).dot(r2) * r2 /2
        n1 = inv(r1)
        n1 *= 1 if n1.y > 0 else -1
        sin_theta = ti.abs(r1.x / 2)
        center = (bricks[i].x + bricks[i].y) / 2
        p1, p2 = boundary_down(sin_theta, n1, bricks[i].x, q_vx[i], center)
        p3, p4 = boundary_down(sin_theta, n1, bricks[i].y, q_vy[i], center)
        p_vx[i] += p1+p4 
        p_vy[i] += p2+p3

        # boundary conditions
        # if Diameter/2 > bricks[i].x.y and q_vx[i].y < 0: 
        #     ra = ti.Vector([bricks[i].x.x, 0.0]) - (bricks[i].x + bricks[i].y) / 2
        #     impact = ti.abs(2 * q_vx[i].y / (1 + 12 / worldl(7) ** 2 * cross_2d(ey, ra)))
        #     p_vx[i] += impact * (ey + sin_theta * 6 * n1)
        #     p_vy[i] += impact * (ey - sin_theta * 6 * n1)
        #     print(impact)
        # if Diameter/2 > bricks[i].y.y and q_vy[i].y < 0: 
        #     ra = ti.Vector([bricks[i].y.x, 0.0]) - (bricks[i].x + bricks[i].y) / 2
        #     impact = ti.abs(2 * q_vy[i].y / (1 + 12 / worldl(7) ** 2 * cross_2d(ey, ra)))
        #     p_vx[i] += impact * (ey - sin_theta * 6 * n1)
        #     p_vy[i] += impact * (ey + sin_theta * 6 * n1)
        #     print(impact)

        for j in range(cnt):
            b1, b2 = contact_ev(i, j), contact_ev(j, i)
            if j != i and (b1 or b2):
        # for j in range(i):
        #     if contact(i,j):
                # default contact_ev, on x
                I, J = i, j 

                if b2 and not b1: 
                    I, J = j, i
                elif b2 and b1 and j < i:   # avoid handling twice
                    I, J = j, i

                t, sign = contact_point_x_or_y(I,J)                
                lam = contact_point_on_edge(I, t)
                r = (bricks[I].y - bricks[I].x).normalized()
                n = inv(r) 
                n *= (1 if n.dot(bricks[J].x - bricks[I].x) > 0 else -1)
                nl = n
                pa = lam * bricks[I].x + (1-lam) * bricks[I].y
                rb1 = bricks[J].x if sign else bricks[J].y
                rbc = (bricks[J].x + bricks[J].y) /2
                rb2 = rbc * 2 - rb1
                rac = (bricks[I].x + bricks[I].y) /2
                ve_0 = lam * v_x[I] + (1 - lam) * v_y[I]
                vv_0 = v_x[J] if sign else v_y[J]
                # FIXME: choice of v_x and q_vx

                # if lam == 0. or lam == 1.:
                if b1 and b2:
                    # handle point-point contact
                    n = (t - pa).normalized()

                v_minus = max(n.dot(ve_0 - vv_0), 0)
                # no need to re-adjust v_minus for point-point contact
                
                ra, rb = pa - rac + n * Diameter/2, -n * Diameter/2 + rb1 - rbc
                impact = ti.abs(2 * v_minus / (2 + 12 / worldl(7) ** 2 * (cross_2d(n, ra) + cross_2d(n, rb))))
                if ti.static(debug) and impact > 0.02:
                    print(f'v- = {v_minus}, impact = {impact}, lam = {lam}, {sign}')
                sin_theta = ti.abs(((rb1 - rbc) / worldl(7)).dot(r))
                n2 = inv((rb1 - rbc).normalized())
                n2 *= 1 if n2.dot(n) > 0 else -1

                # px = q_vx[i] + (q_vy[i] - q_vx[i]).dot(r1) * r1 /2
                # py = q_vy[i] - (q_vy[i] - q_vx[i]).dot(r1) * r1 /2
                # px = px0
                # py = py0
                px = py = ti.Vector.zero(float, 2)
                if (b1 and not b2) or (b1 and b2 and i < j):
                    px += impact * (-n + (lam - 0.5) * 6 * nl)
                    py += impact * (-n - (lam - 0.5) * 6 * nl)
                elif (b2 and not b1) or (b1 and b2 and i > j):
                    if sign:
                        px += impact * (n + sin_theta * 6 * n2)
                        py += impact * (n - sin_theta * 6 * n2)
                    else :
                        px += impact * (n - sin_theta * 6 * n2) 
                        py += impact * (n + sin_theta * 6 * n2)

                p_vx[i] += px
                p_vy[i] += py



@ti.kernel
def global_v(cnt: ti.i32):
    for i in bricks:
        q_vx[i] = p_vx[i] / max(contacts[i] - 1, 1)
        q_vy[i] = p_vy[i] / max(contacts[i] - 1, 1)


@ti.kernel
def minimize_global(cnt: ti.i32):
    for i in bricks:
        t1, t2 = bricks[i].x, bricks[i].y
        bricks[i].x += p_x[i]/ contacts[i]
        bricks[i].y += p_y[i]/ contacts[i]

        bricks[i].x *= contacts[i]/(contacts[i]+1)
        bricks[i].y *= contacts[i]/(contacts[i]+1)

        bricks[i].x += s_x[i] /(contacts[i] +1) 
        bricks[i].y += s_y[i] /(contacts[i] +1) 
        t3, t4 = bricks[i].x, bricks[i].y
        if i == cnt-1 and debug:
            print(f'residual :{(t3-t1).norm()}, {(t4-t2).norm()}, contacts: {contacts[i]}')
        
def solve_local_global(cnt): 
    copy_x_to_s(s_x, s_y)
    for i in range(n_jacobian_iters):
        if debug:
            print(f'iter{i}')
        project_local(cnt)
        minimize_global(cnt)

def solve_v(cnt):
    q_vx.copy_from(v_x)
    q_vy.copy_from(v_y)
    for i in range(n2_jacobian_iters):
        project_v(cnt)
        global_v(cnt)
    v_x.copy_from(q_vx)
    v_y.copy_from(q_vy)
# @ti.kernel
# def collision_projection(i: ti.i32):
#     for j in range(cnt):
#         pass
#         # if xc(j):

        
@ti.kernel
def check_fenwick(I:ti.i32, cnt: ti.i32) -> ti.i32:
    ret = 0
    for i in range(cnt):
        if i != I and tmp_dense_matrix[i,I]:
            ret = 1
    return ret

@ti.kernel
def add_gravity(g: ti.i32):
    for i in bricks:
        bricks[i].x += v_x[i] * dt
        bricks[i].y += v_y[i] * dt
        if g:
            v_x[i] += dt * ti.Vector([0.0, -500.])
            v_y[i] += dt * ti.Vector([0.0, -500.])
        
@ti.kernel
def update_velocity():
    for i in bricks:
        v_x[i] = (bricks[i].x - q_x[i]) / dt
        v_y[i] = (bricks[i].y - q_y[i]) / dt

def probe_grid_sdf(cnt):
    # with ti.Tape(tot_sdf):
    #     grid_sdf(cnt)
    test_wrapper(cnt)
    _grid = probes.to_numpy().reshape((-1,2))
    _grad = probes.grad.to_numpy().reshape((-1,2)) / 40.0
    return _grid, _grad
    # print(_grad)

# window = ti.ui.Window('Implicit Mass Spring System', res=(500, 500))
gui = ti.GUI("Vector Field", res=(res, res))
        
cnt = init()
_grid, _grad = probe_grid_sdf(0)
mxy = np.zeros((2,2),dtype = np.float32)
mcnt = 0
normalize = lambda x: x/np.linalg.norm(x)
t3 = False
pause = False
gravity = False
to_np = lambda x,y: bricks.get_member_field(x).to_numpy()[:y+1]

while gui.running:
    
    l,x,y = [to_np(i,cnt-mcnt) for i in ['l','x','y']]
    if len(l):
        gui.lines(x, y, color = 0x888888, radius = radius)
        gui.circles(np.vstack([x, y]), radius = Radius, color = 0x0)
        v = np.vstack([v_x.to_numpy()[:cnt-mcnt + 1], v_y.to_numpy()[:cnt-mcnt + 1]]) / 100.0
        gui.arrows(np.vstack([x, y]), v, radius = 3, color= 0x888800)
    
    gui.circles(_grid, radius=3)
    gui.arrows(_grid, _grad, radius = 1)
    if not pause:
        copy_x_to_s(q_x,q_y)
        add_gravity(gravity)
        solve_local_global(cnt)
        solve_v(cnt)
        # update_velocity()
    if gui.get_event(ti.GUI.PRESS):
        e = gui.event
        print(e.key)
        if e.key == 'r':
            container.deactivate_all()
            cnt = init()
        elif e.key == 's':
            pause = not pause
        elif e.key == 'g':
            gravity = not gravity
        elif e.key == ti.GUI.LMB:
            mxy = np.array(gui.get_cursor_pos(), dtype=np.float32) 
            mcnt = not mcnt if not (t3 and not allow_cross) else mcnt
            print(mxy, mcnt,cnt)
            if mcnt:
                bricks[cnt] = ti.Struct({
                    'x': ti.Vector(mxy), # center
                    'y': ti.Vector(mxy), # center
                    'l': 7,  # length
                })
            elif not (t3 and not allow_cross):
                _grid, _grad = probe_grid_sdf(cnt)
                cnt += 1      
                solve_local_global(cnt)
                
    elif mcnt :
        m = np.array(gui.get_cursor_pos())
        preview_brick(m-mxy,cnt)
        tmp_dense_matrix.fill(0)
        ev(cnt)
        t1 = check_fenwick(cnt, cnt)
        ve(cnt)
        t2 = check_fenwick(cnt, cnt)
        ee(cnt)
        t3 = check_fenwick(cnt, cnt)
        
        color = 0x880000 if t3 else 0x440000 if t1 or t2 else 0x444444 
        l,x,y = bricks[cnt].l, bricks[cnt].x, bricks[cnt].y
        gui.line(x, y, color = color, radius = radius)
        gui.circle(x, color = 0x0, radius = Radius)
    gui.show()

# while window.running:
#     if window.get_event(ti.ui.PRESS):
#         if window.event.key == ti.ui.ESCAPE:
#             break
#     ti.ui.arrows(grid.to_numpy(), grid.grad.to_numpy(), radius = radius)
    # if window.is_pressed(ti.ui.SPACE):
    #     pause = not pause

    # if not pause:
    #     cloth.preview_brick(h)

    # canvas = window.get_canvas()
    # cloth.displayGGUI(canvas)
    # window.show()

