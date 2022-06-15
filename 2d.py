import taichi as ti
import numpy as np 
# (x|y|r)\[(.)\]    bricks[$2].$1

dim = 2
max_bricks, max_joints = 1000, 500
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
epsilon = 1. + 0.
dt = 5e-4
debug = False
debug_v = False
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
joints = ti.Vector.field(2, dtype = ti.i32)
joints_lambda = ti.Vector.field(2, dtype = float)
joints_container = ti.root.pointer(ti.i, 10).pointer(ti.i, 10).pointer(ti.i, max_joints // 100)
joints_container.place(joints, joints_lambda)
bricks = ti.StructField(linear_dict)
container = ti.root.pointer(ti.i, 10).pointer(ti.i, 10).pointer(ti.i, max_joints // 100)
container.place(bricks)
container.place(p_x, p_y,p_vx, p_vy, s_x, s_y, v_x, v_y, q_x, q_y, q_vx, q_vy, contacts)
probes = ti.Vector.field(2, float, (grid,grid))
sdf = ti.field(float, (grid, grid))
tot_sdf = ti.field(float, ())
tmp_dense_matrix = ti.field(ti.u8, (100,100))
# temporary joint info 
joints_preview_info = ti.Vector.field(2, dtype = ti.i32, shape = ())
completed_joint_number = ti.field(dtype = ti.i32, shape = ())
jn = ti.field(dtype = ti.i32, shape = ())
lambda_preview_info = ti.Vector.field(2, dtype = float, shape = ())


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
def init_case_3() -> ti.i32:    # for testing joints
    for I in ti.grouped(probes):
        probes[I] = (I + 0.5) / grid
    x2 = ti.Vector([0.1, 0.2])
    y2 = x2 + ey * worldl(7)
    
    x3 = x2 + ey * worldl(5)
    y3 = x3 + ex * worldl(7)
    bricks[0] = ti.Struct({
        'x': x2, 
        'y': y2,
        'l': 7,  # length
    })
    bricks[1] = ti.Struct({
        'x': x3, 
        'y': y3,
        'l': 7,  # length
    })
    joints[0] = ti.Vector([0,1])
    joints_lambda[0] = ti.Vector([1/3, 1.])
    # print(f'jointed(0,1) = {jointed(0,1)}')
    v_x[0] = v_y[0] = ey * 5
    v_x[1] = v_y[1] = -ey * 5
    return 2

        
@ti.kernel
def init_case_2() -> ti.i32:    # for testing point-point contact
    for I in ti.grouped(probes):
        probes[I] = (I + 0.5) / grid
    x2 = ti.Vector([0.1, 0.2])
    y2 = x2 + ey * worldl(7)
    x3 = ti.Vector([0.5, y2.y + Diameter/4])
    y3 = x3 + ti.Vector([1., -0.5]).normalized() * worldl(7)
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
def init_case_1() -> ti.i32:    # for testing regular case
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
def mingle(i, lam):
    return bricks[i].x * lam + bricks[i].y * (1- lam)

@ti.func
def interpolate(x, i):
    n = bricks[i].l - 1.
    lam = contact_point_on_edge(i, x)
    _lam = ti.cast(ti.round(lam * n), float) / n
    return bricks[i].x * _lam + bricks[i].y * (1-_lam), _lam

@ti.kernel
def preview_neighbouring_joint(arr: ti.ext_arr(), ret: ti.ext_arr()) -> ti.i32:
    x = ti.Vector([arr[0], arr[1]])
    hole = ti.Vector.zero(float, 2)
    b = 0
    for i in bricks:
        if signed_distance(x, i) < Diameter /2:
            hole, lam = interpolate(x, i)
            ret[0], ret[1] = i, lam
            b = 1
    arr[0], arr[1] = hole.x, hole.y
    return b

@ti.kernel
def add_joint(cnt: ti.i32, arr: ti.ext_arr()):
    j = ti.Vector([ti.cast(arr[0], ti.i32), -1])
    jl = ti.Vector([arr[1], -1.])
    # joints_pos[cnt] = x
    joints[cnt], joints_lambda[cnt] = j, jl
    
    # joints[cnt] = ti.Vector([i, j])
    # joints_lambda[cnt] = ti.Vector([l1, l2])

@ti.kernel
def get_joints_pos(arr: ti.ext_arr()):
    for j in joints:
        x = mingle(joints[j].x, joints_lambda[j].x)
        arr[j, 0], arr[j, 1] = x.x, x.y

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

@ti.func
def _ee(i, j):
    r1 = bricks[i].y - bricks[i].x
    r2 = bricks[j].y - bricks[j].x
    m11 = ti.Matrix.cols([bricks[j].x-bricks[i].x, r1])
    m12 = ti.Matrix.cols([bricks[j].y-bricks[i].y, r1])
    m21 = ti.Matrix.cols([bricks[j].x-bricks[i].x, r2])
    m22 = ti.Matrix.cols([bricks[j].y-bricks[i].y, r2])
    t = m11.determinant() * m12.determinant() < 0 and m21.determinant() * m22.determinant() < 0
    return t

@ti.kernel
def ee(i:ti.i32):
    for j in bricks:
        t = _ee(i, j)
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
            if j != I and contact(I,j) and not jointed(I, j):
                if contact_ev(I,j):
                    contact_point, _ = contact_point_x_or_y(I,j)
                    p_x[I] -= (Diameter - signed_distance(contact_point, I)) * nebla_sdf(contact_point, I).normalized()
                    p_y[I] -= (Diameter - signed_distance(contact_point, I)) * nebla_sdf(contact_point, I).normalized()
                else:   # contact ve
                    p_x[I] += max(-signed_distance(bricks[I].x, j) + Diameter, 0.) * nebla_sdf(bricks[I].x, j).normalized()
                    p_y[I] += max(-signed_distance(bricks[I].y, j) + Diameter, 0.) * nebla_sdf(bricks[I].y, j).normalized()
                contacts[I] += 1

@ti.func
def jointed(b1, b2):
    ret = 0
    for j in range(n_joints):   # avoid using struct-for in func
        I, J = joints[j].x, joints[j].y 
        if (I == b1 and J == b2) or (I == b2 and J == b1):
            ret += 1
            jn[None] = j
    return ret

@ti.kernel
def complete_joint():
    j = completed_joint_number[None]
    joints[j] = joints_preview_info[None]
    joints_lambda[j] = lambda_preview_info[None]

@ti.kernel
def solve_joints():
    for j in joints:
        if joints[j].y != -1:
            i1, i2 = joints[j].x, joints[j].y
            x1 = mingle(i1, joints_lambda[j].x)
            x2 = mingle(i2, joints_lambda[j].y)
            p_x[i1] += x2 - x1
            p_y[i1] += x2 - x1
            p_x[i2] += x1 - x2
            p_y[i2] += x1 - x2
            contacts[i1] += 1
            contacts[i2] += 1

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
def boundary_down(r1, n1, x, q_v, center):
    impact = 0.
    sin_theta = ti.abs(r1.x / 2)
    if Diameter/2 > x.y and q_v.y < 0: 
        ra = ti.Vector([x.x, 0.0]) - center
        impact = ti.abs(epsilon * q_v.y / (1 + 12 / worldl(7) ** 2 * cross_2d(ey, ra)))
    return impact * (ey + sin_theta * 6 * n1), impact * (ey - sin_theta * 6 * n1)
    
@ti.func
def boundary_left(r1, n1, x, q_v, center):
    impact = 0.
    sin_theta = ti.abs(r1.y / 2)
    if Diameter/2 > x.x and q_v.x < 0: 
        ra = ti.Vector([0.0, x.y]) - center
        impact = ti.abs(epsilon * q_v.x / (1 + 12 / worldl(7) ** 2 * cross_2d(ex, ra)))
    return impact * (ex + sin_theta * 6 * n1), impact * (ex - sin_theta * 6 * n1)

@ti.func
def boundary_right(r1, n1, x, q_v, center):
    impact = 0.
    sin_theta = ti.abs(r1.y / 2)
    if Diameter/2 > 1.0 - x.x and q_v.x > 0: 
        ra = ti.Vector([1.0, x.y]) - center
        impact = ti.abs(epsilon * q_v.x / (1 + 12 / worldl(7) ** 2 * cross_2d(-ex, ra)))
    return impact * (-ex + sin_theta * 6 * n1), impact * (-ex - sin_theta * 6 * n1)

@ti.kernel
def project_v(cnt: ti.i32, v_x:ti.template(), v_y: ti.template(), q_vx: ti.template(), q_vy: ti.template()):
    for i in bricks:
        r1 = (bricks[i].y - bricks[i].x).normalized()
        p_vx[i] = p_vy[i] = ti.Vector.zero(float, 2)

        n1 = inv(r1)
        n2 = n1 * (1 if n1.x > 0 else -1)
        n3 = -n2
        n1 *= 1 if n1.y > 0 else -1

        center = (bricks[i].x + bricks[i].y) / 2

        p11, p12 = boundary_down(r1, n1, bricks[i].x, q_vx[i], center)
        p13, p14 = boundary_down(r1, n1, bricks[i].y, q_vy[i], center)

        p11, p12 = boundary_down(r1, n1, bricks[i].x, q_vx[i], center)
        p13, p14 = boundary_down(r1, n1, bricks[i].y, q_vy[i], center)

        p21, p22 = boundary_left(r1, n2, bricks[i].x, q_vx[i], center)
        p23, p24 = boundary_left(r1, n2, bricks[i].y, q_vy[i], center)

        p31, p32 = boundary_right(r1, n3, bricks[i].x, q_vx[i], center)
        p33, p34 = boundary_right(r1, n3, bricks[i].y, q_vy[i], center)

        p_vx[i] += p11+p14 + p21+p24 + p31+p34  
        p_vy[i] += p12+p13 + p22+p23 + p32+p33 

        for j in range(cnt):
            b1, b2, b3 = contact_ev(i, j), contact_ev(j, i), jointed(i, j)
            if (j != i and (b1 or b2)) or b3:
        
                # default contact_ev, on x
                I, J = i, j 
                cj = 0
                # FIXME: should make it jn[None] but bug in `jointed` func  
                if b3:
                    I, J = joints[cj].x, joints[cj].y
                elif b2 and not b1: 
                    I, J = j, i
                elif b2 and b1 and j < i:   # avoid handling twice
                    I, J = j, i
                    # so far I < J or J point-contact on I's edge

                t, sign = contact_point_x_or_y(I,J)                
                lam = contact_point_on_edge(I, t) if not b3 else joints_lambda[cj].x
                lam2 = joints_lambda[cj].y
                '''
                b3 content explained : lazy implementation, 
                    use the oritation of relative v as n, and epsilon = 0,
                    such that the bricks sticks together at joints 
                '''
                ve_0 = lam * v_x[I] + (1 - lam) * v_y[I]
                vv_0 = lam2 * v_x[J] + (1 - lam2) * v_y[J] if b3 else v_x[J] if sign else v_y[J]
                # FIXME: choice of v_x and q_vx

                r = (bricks[I].y - bricks[I].x).normalized()
                n = inv(r) 
                n *= (1 if n.dot(bricks[J].x - bricks[I].x) > 0 else -1)
                if b3 :
                    n = -((ve_0 - vv_0).normalized()) 
                nl = n
                pa = lam * bricks[I].x + (1-lam) * bricks[I].y
                pb = lam2 * bricks[J].x + (1-lam2) * bricks[J].y
                rb1 = pb if b3 else bricks[J].x if sign else bricks[J].y
                rbc = (bricks[J].x + bricks[J].y) /2
                # rb2 = rbc * 2 - rb1
                rac = (bricks[I].x + bricks[I].y) /2

                # if lam == 0. or lam == 1.:
                if not b3 and b1 and b2:
                    # handle point-point contact
                    n = (t - pa).normalized()

                v_minus = max(n.dot(ve_0 - vv_0), 0) if not b3 else (ve_0-vv_0).norm()
                # no need to re-adjust v_minus for point-point contact
                
                ra, rb = pa - rac + n * Diameter/2, -n * Diameter/2 + rb1 - rbc 
                if b3:
                    ra, rb = pa - rac, rb1 - rbc
                # epsilon = 1. if not b3 else 0.
                impact = ti.abs(1 * v_minus / (2 + 12 / worldl(7) ** 2 * (cross_2d(n, ra) + cross_2d(n, rb))))
                if ti.static(debug_v) and impact > 0.02:
                    print(f'v- = {v_minus}, impact = {impact}, lam = {lam}, {sign}, n = [{n[0]},{n[1]}]')
                # sin_theta = ti.abs(((rb1 - rbc) / worldl(7)).dot(r)) if not (b1 and b2) else ti.abs(n.cross((rb1 - rbc) / worldl(7)))
                sin_theta = ti.abs(n.cross(rb1 - rbc)) / worldl(7)
                sin_theta_2 = ti.abs(n.cross(pa - rac)) / worldl(7)
                n2 = inv((rb1 - rbc).normalized())
                n2 *= 1 if n2.dot(n) > 0 else -1

                px = py = ti.Vector.zero(float, 2)
                if b3: 
                    t1 = t2 = ti.Vector.zero(float, 2)
                    if i == I:
                        t1 = impact * (n + (lam - 0.5) * 6 * (n - n.dot(ra) * ra / ra.norm_sqr()))
                        t2 = impact * (n - (lam - 0.5) * 6 * (n - n.dot(ra) * ra / ra.norm_sqr()))
                    else:

                        t1 = -impact * (n + (lam2 - 0.5) * 6 * (n - n.dot(rb) * rb / rb.norm_sqr()))
                        t2 = -impact * (n - (lam2 - 0.5) * 6 * (n - n.dot(rb) * rb / rb.norm_sqr()))
                    px += t1
                    py += t2
                    if ti.static(debug_v):
                        print(f'i = {i}, px, py = [{t1[0]},{t1[1]}], [{t2[0]},{t2[1]}]')

                elif (b1 and not b2): # or (b1 and b2 and i < j):
                    px += impact * (-n - (lam - 0.5) * 6 * nl)
                    py += impact * (-n + (lam - 0.5) * 6 * nl)
                elif b1 and b2 and i < j:
                    px += impact * (-n - sin_theta_2 * 6 * nl)
                    py += impact * (-n + sin_theta_2 * 6 * nl)
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
        r1 = (bricks[i].y - bricks[i].x).normalized()
        px0 = q_vx[i] + (q_vy[i] - q_vx[i]).dot(r1) * r1 /2
        py0 = q_vy[i] - (q_vy[i] - q_vx[i]).dot(r1) * r1 /2
        # p_vx[i] += px0, p_vy[i] = px0, py0
        q_vx[i] = p_vx[i] / max(contacts[i] - 1, 1) + px0
        q_vy[i] = p_vy[i] / max(contacts[i] - 1, 1) + py0


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
        solve_joints()
        minimize_global(cnt)

def solve_v(cnt):
    q_vx.copy_from(v_x)
    q_vy.copy_from(v_y)
    for i in range(n2_jacobian_iters):
        project_v(cnt, v_x, v_y, q_vx, q_vy)
        global_v(cnt)
    v_x.copy_from(q_vx)
    v_y.copy_from(q_vy)
        
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
        _g = -ey if g == 1 else ex if g == 2 else -ex if g == 3 else ti.Vector.zero(float, 2)
        v_x[i] += dt * 500. * _g 
        v_y[i] += dt * 500. * _g

        
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

@ti.kernel
def preview_possible_assemble(m: ti.ext_arr(), mxy: ti.ext_arr(), ret: ti.ext_arr(), cnt: ti.i32)->ti.i32:
    mouse = ti.Vector([m[0], m[1]])
    bx = ti.Vector([mxy[0], mxy[1]])
    b = 0
    for j in joints:
        if joints[j].y == -1 and b == 0:
            x = mingle(joints[j].x, joints_lambda[j].x)
            if (x-mouse).norm() < Diameter/2:
                b += 1
                ret[0], ret[1] = x.x, x.y


                r = (x - bx).normalized()
                bricks[cnt].y = r * worldl(bricks[cnt].l) + bricks[cnt].x

                completed_joint_number[None] = j
                joints_preview_info[None] = joints[j]
                lambda_preview_info[None] = joints_lambda[j]


                joints_preview_info[None].y = cnt
                hole, lam = interpolate(x, cnt)
                lambda_preview_info[None].y = lam
                ''' FIXME: 
                1. adjust position along the brick; 
                2. if e-v contact after the move, warning color
                3. invalidate e-e contact with other bricks
                4. global solve after confirming the assemble

                '''
    return b

# window = ti.ui.Window('Implicit Mass Spring System', res=(500, 500))
gui = ti.GUI("LEGO master breaker", res=(res, res))

CASE = 2
INIT_CASES = [init_case_1, init_case_2, init_case_3]
N_JOINTS_LIST = [0, 0, 1]
init, n_joints = INIT_CASES[CASE], N_JOINTS_LIST[CASE]

cnt = init()
_grid, _grad = probe_grid_sdf(0)
mxy = np.zeros((2,2),dtype = np.float32)
mcnt = 0
normalize = lambda x: x/np.linalg.norm(x)
t3 = False
pause = False
gravity = False
to_np = lambda x,y: bricks.get_member_field(x).to_numpy()[:y+1]
BRICKS_MODE = 0
JOINT_MODE = 1
mode = BRICKS_MODE
tmp_joint = np.zeros((2,), dtype = np.float32)
new_joint_allowed = False
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
        if e.key == 'r' or e.key =='c':
            container.deactivate_all()
            joints_container.deactivate_all()
            cnt = init() if e.key == 'r' else 0
            n_joints = N_JOINTS_LIST[CASE]
        elif e.key == 's':
            pause = not pause
        elif e.key == 'g':
            gravity = not gravity
        elif e.key == ti.GUI.LEFT:
            gravity = 3
        elif e.key == ti.GUI.RIGHT:
            gravity = 2
        elif e.key == ti.GUI.DOWN:
            gravity = 1
        elif e.key == 'f':
            mode = not mode
        elif e.key == ti.GUI.LMB:
            mxy = np.array(gui.get_cursor_pos(), dtype=np.float32) 
            if mode == BRICKS_MODE:    
                mcnt = not mcnt if not (t3 and not allow_cross) else mcnt
                print(mxy, mcnt,cnt)
                if mcnt:
                    bricks[cnt] = ti.Struct({
                        'x': ti.Vector(mxy), # center
                        'y': ti.Vector(mxy), # center
                        'l': 7,  # length
                    })
                elif not (t3 and not allow_cross):
                    if picked:
                        complete_joint()
                    _grid, _grad = probe_grid_sdf(cnt)
                    cnt += 1      
                    solve_local_global(cnt)
            elif mode == JOINT_MODE and new_joint_allowed: # joint mode
                add_joint(n_joints, tmp_joint)
                n_joints += 1
                # print(n_joints)
                # print(joints.to_numpy()[:n_joints], joints_lambda.to_numpy()[:n_joints])
                
    elif mcnt or mode == JOINT_MODE:
        m = np.array(gui.get_cursor_pos())
        if mcnt:
            preview_brick(m-mxy,cnt)
            tmp_dense_matrix.fill(0)
            # tj = ej(cnt)
            picked = preview_possible_assemble(m, mxy, tmp_joint, cnt)
            ev(cnt)
            t1 = check_fenwick(cnt, cnt)
            ve(cnt)
            t2 = check_fenwick(cnt, cnt)
            ee(cnt)
            t3 = check_fenwick(cnt, cnt) and not picked
            
            color = 0x880000 if t3 else 0x440000 if (t1 or t2 and not picked) else 0x444444 
            l,x,y = bricks[cnt].l, bricks[cnt].x, bricks[cnt].y
            gui.line(x, y, color = color, radius = radius)
            gui.circle(x, color = 0x0, radius = Radius)
            if picked:
                gui.circle(tmp_joint, color = 0xffffff, radius = Radius * 0.9)
        else:
            new_joint_allowed = preview_neighbouring_joint(m, tmp_joint)
            # tmp_joint = m
            gui.circle(m, color = 0xffffff, radius = Radius * 0.8)
        
    if n_joints > 0:
        jx = np.zeros((n_joints, 2), np.float32)
        get_joints_pos(jx)
        gui.circles(jx, color = 0xffffff, radius = Radius * 0.8)
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

