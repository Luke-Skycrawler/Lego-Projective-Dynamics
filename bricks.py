import taichi as ti
def Bricks(globals, signed_distance, inv, contact_ev, contact, jointed, nebla_sdf, contact_point_on_edge):
    dim = globals.dim
    max_bricks = globals.max_bricks
    max_joints = globals.max_joints
    chunk = globals.chunk
    res = globals.res
    grid = globals.grid
    radius = globals.radius
    Radius = globals.Radius
    diameter = globals.diameter
    zero = globals.zero
    Diameter = globals.Diameter
    n_jacobian_iters = globals.n_jacobian_iters
    n2_jacobian_iters = globals.n2_jacobian_iters
    epsilon = globals.epsilon
    dt = globals.dt
    debug = globals.debug
    debug_v = globals.debug_v
    allow_cross = globals.allow_cross
    flight = globals.flight
    ex = globals.ex
    ey = globals.ey
    contacts = globals.contacts
    p_x = globals.p_x
    p_y = globals.p_y
    p_vx = globals.p_vx
    p_vy = globals.p_vy
    s_x = globals.s_x
    s_y = globals.s_y
    v_x = globals.v_x
    v_y = globals.v_y
    q_x = globals.q_x
    q_y = globals.q_y
    q_vx = globals.q_vx
    q_vy = globals.q_vy
    joints = globals.joints
    joints_lambda = globals.joints_lambda
    bricks = globals.bricks
    jn = globals.jn
    worldl = lambda l: (l-1) * Diameter

    @ti.func
    def contact_point_x_or_y(I,j):  # return 1 for x 0 for y
        boolean = signed_distance(bricks[j].x, I) < signed_distance(bricks[j].y, I)
        contact_point = bricks[j].x if boolean else bricks[j].y
        return contact_point, boolean
        
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
    def project_v(cnt: ti.i32, n_joints: ti.i32, v_x:ti.template(), v_y: ti.template(), q_vx: ti.template(), q_vy: ti.template()):
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
                b1, b2, b3 = contact_ev(i, j), contact_ev(j, i), jointed(i, j, n_joints)
                if (j != i and (b1 or b2)) or b3:
            
                    # default contact_ev, on x
                    I, J = i, j 
                    cj = jn[i]
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
                    v_rel = ve_0 - vv_0
                    # FIXME: choice of v_x and q_vx

                    r = (bricks[I].y - bricks[I].x).normalized()
                    n = inv(r) 
                    n *= (1 if n.dot(bricks[J].x - bricks[I].x) > 0 else -1)
                    if b3 and v_rel.norm() > zero:
                        n = -(v_rel.normalized()) 
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

                    v_minus = max(n.dot(v_rel), 0) if not b3 else v_rel.norm()
                    # no need to re-adjust v_minus for point-point contact
                    
                    ra, rb = pa - rac + n * Diameter/2, -n * Diameter/2 + rb1 - rbc 
                    if b3:
                        ra, rb = pa - rac, rb1 - rbc
                    # epsilon = 1. if not b3 else 0.
                    impact = ti.abs(1 * v_minus / (2 + 12 / worldl(7) ** 2 * (cross_2d(n, ra) + cross_2d(n, rb))))
                    if ti.static(debug_v) and impact > 0.02:
                        print(f'v- = {v_minus}, impact = {impact}, lam = {lam}, {sign}, n = [{n[0]},{n[1]}]')
                    sin_theta = ti.abs(n.cross(rb1 - rbc)) / worldl(7)
                    sin_theta_2 = ti.abs(n.cross(pa - rac)) / worldl(7)
                    n2 = inv((rb1 - rbc).normalized())
                    n2 *= 1 if n2.dot(n) > 0 else -1

                    px = py = ti.Vector.zero(float, 2)
                    if b3: 
                        rot = t1 = t2 = ti.Vector.zero(float, 2)
                        _r, _lam = ra, lam 
                        if i != I:
                            _r, _lam = rb, lam2
                        if _r.norm() > zero:
                            rot = (_lam - 0.5) * 6 * (n - n.dot(_r) * _r / _r.norm_sqr())
                        if i == I:
                            t1 = impact * (n + rot)
                            t2 = impact * (n - rot)
                        else:
                            t1 = -impact * (n + rot)
                            t2 = -impact * (n - rot)

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
    def project_local(cnt: ti.i32, n_joints: ti.i32):   # local step for all bricks 

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
                if j != I and contact(I,j) and not jointed(I, j, n_joints):
                    if contact_ev(I,j):
                        contact_point, _ = contact_point_x_or_y(I,j)
                        p_x[I] -= (Diameter - signed_distance(contact_point, I)) * nebla_sdf(contact_point, I).normalized()
                        p_y[I] -= (Diameter - signed_distance(contact_point, I)) * nebla_sdf(contact_point, I).normalized()
                    else:   # contact ve
                        p_x[I] += max(-signed_distance(bricks[I].x, j) + Diameter, 0.) * nebla_sdf(bricks[I].x, j).normalized()
                        p_y[I] += max(-signed_distance(bricks[I].y, j) + Diameter, 0.) * nebla_sdf(bricks[I].y, j).normalized()
                    contacts[I] += 1
    
    @ti.kernel
    def preview_brick(arr:ti.ext_arr(), cnt: ti.i32):
        r = ti.Vector([arr[0],arr[1]]).normalized()
        bricks[cnt].y = r * worldl(bricks[cnt].l) + bricks[cnt].x                    

    return project_v, project_local, preview_brick
