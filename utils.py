import taichi as ti 

    
def Utils(globals):
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
    ex = globals.ex
    ey = globals.ey
    bricks = globals.bricks
    probes = globals.probes
    sdf = globals.sdf
    tot_sdf = globals.tot_sdf
    tmp_dense_matrix = globals.tmp_dense_matrix
    jn = globals.jn
    joints = globals.joints

    worldl = lambda l: (l-1) * Diameter
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
    def grid_sdf(t: ti.i32):
        for I in ti.grouped(sdf):
            sdf[I] = signed_distance(probes[I], t)
            tot_sdf[None] += sdf[I]

    @ti.kernel
    def test_wrapper(cnt: ti.i32):
        for i in ti.grouped(probes):
            probes.grad[i] = nebla_sdf(probes[i], cnt)

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

    def probe_grid_sdf(cnt):
        # with ti.Tape(tot_sdf):
        #     grid_sdf(cnt)
        test_wrapper(cnt)
        _grid = probes.to_numpy().reshape((-1,2))
        _grad = probes.grad.to_numpy().reshape((-1,2)) / 40.0
        return _grid, _grad
        # print(_grad)

    @ti.func
    def contact_point_on_edge(i, contact_point):                
        # contact point =  lam * x + (1 - lam) * y
        lam = 1. - (contact_point - bricks[i].x).dot((bricks[i].y - bricks[i].x).normalized())/worldl(7)
        return max(min(lam , 1.), 0.)

    @ti.func
    def jointed(b1, b2, n_joints):
        ret = 0
        for j in range(n_joints):   # avoid using struct-for in func
            I, J = joints[j].x, joints[j].y 
            if (I == b1 and J == b2) or (I == b2 and J == b1):
                ret += 1
                jn[b1] = j
        return ret

    return inv, R, signed_distance, grid_sdf, test_wrapper, update_fenwick, contact, contact_ev, ve, ev, _ee, ee, nebla_sdf, copy_x_to_s, probe_grid_sdf, contact_point_on_edge, jointed
