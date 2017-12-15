#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>


import os
import math
import numpy as np
import solvcon as sc


def create_sample_block():
    blk = sc.Block(ndim=2, nnode=8, ncell=7, nbound=7)
    blk.ndcrd[...] = (
        (0  , 0  ),
        (1  , 0  ),
        (3  , 0  ),
        (3  , 2  ),
        (1.2, 2  ),
        (0  , 2  ),
        (0  , 0.9),
        (2  , 1.1),
    )
    blk.cltpn[...] = 3
    blk.clnds[:,:4] = (
        (3, 3,4,7),
        (3, 4,5,6),
        (3, 1,7,6),
        (3, 4,6,7),
        (3, 0,1,6),
        (3, 1,2,7),
        (3, 2,3,7),
    )
    blk.build_interior()
    blk.build_boundary()
    blk.build_ghost()
    return blk


class FlowControl(object):

    def __init__(self, obj, **kw):
        self.obj = obj
        self.func = kw.pop("func")
        self.require = kw.pop("require", list())
        self.require_not = kw.pop("require_not", list())
        self.executed = False

    def __call__(self, *args, **kw):
        if self.executed:
            raise RuntimeError("this method can be called only once")
        for name in self.require:
            method = getattr(self.obj, name)
            if not method.executed:
                raise RuntimeError("%s isn't called yet" % method.func.__name__)
        for name in self.require_not:
            method = getattr(self.obj, name)
            if method.executed:
                raise RuntimeError("%s has been called" % method.func.__name__)
        ret = self.func(self.obj, *args, **kw)
        self.executed = True
        return ret


class FlowControlDescriptor(object):

    def __init__(self, **kw):
        self.kw = kw

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        wrapper = FlowControl(obj, **self.kw)
        setattr(obj, wrapper.func.__name__, wrapper) # replace.
        return wrapper


def flow_control(**kw):
    def wrapper(func):
        return FlowControlDescriptor(func=func, **kw)
    return wrapper


class PsPicture(list):

    def __init__(self, *args, **kw):
        self.blk = kw.pop("blk")
        self.unit = kw.pop("unit")
        self.x0 = kw.pop("x0")
        self.y0 = kw.pop("y0")
        self.x1 = kw.pop("x1")
        self.y1 = kw.pop("y1")
        super(PsPicture, self).__init__(*args, **kw)

    @flow_control(require_not=["add_footer"])
    def add_header(self):
        self.append("\\psset{unit=%s}" % self.unit)
        self.append("\\begin{pspicture}(%g,%g)(%g,%g)" % 
                    (self.x0, self.y0, self.x1, self.y1))
        self.append("")

    @flow_control(require=["add_header"])
    def add_footer(self):
        self.append("\\end{pspicture}")

    @staticmethod
    def build_angle_map(blk):
        angle_map = dict()
        for fc in blk.fcnds:
            crd0 = blk.ndcrd[fc[1]]
            crd1 = blk.ndcrd[fc[2]]
            angle = math.atan2(crd1[1]-crd0[1], crd1[0]-crd0[0])
            angle_map.setdefault(fc[1], list()).append(angle)
            angle = math.atan2(crd0[1]-crd1[1], crd0[0]-crd1[0])
            angle_map.setdefault(fc[2], list()).append(angle)
        for nd in angle_map:
            angles = sorted(angle_map[nd])
            sectors = [[a1-a0, a0, a1] for a0, a1 in
                       zip(angles, angles[1:] + [angles[0]])]
            sectors[-1][0] += math.pi * 2 # make it positive.
            # pick the largest included angle.
            idx = np.array([sector[0] for sector in sectors]).argmax()
            sector = sectors[idx]
            angle = (sector[1] + sector[2]) / 2
            # fix the wrapped sector.
            if sector[0] > math.pi*(1-1.e-15) and idx == len(sectors)-1:
                angle += math.pi
            angle *= 180 / math.pi
            angle_map[nd] = angle
        return angle_map

    @flow_control(require=["add_header"], require_not=["add_footer"])
    def add_nodes(self, label=False):
        for ind, crd in enumerate(self.blk.ndcrd):
            self.append("\\dotnode[dotstyle=*,dotsize=4pt](%g,%g){nd%d}" %
                        (crd[0], crd[1], ind))
        if label:
            angle_map = self.build_angle_map(self.blk)
            for ind in range(self.blk.nnode):
                self.append("\\nput[labelsep=3pt]{%.0f}{nd%d}{\\texttt{%d}}" %
                            (angle_map[ind], ind, ind))
        self.append("")
        self._has_nodes = True

    @flow_control(require=["add_nodes"], require_not=["add_footer"])
    def add_faces(self):
        for fc in self.blk.fcnds:
            self.append("\\ncline[linewidth=0.5pt]{nd%d}{nd%d}" % tuple(fc[1:3]))
        self.append("")
        self._has_faces = True

    @flow_control(require=["add_header"], require_not=["add_footer"])
    def add_cell_centers(self):
        for icl, cnd in enumerate(self.blk.clcnd):
            self.append("\\dotnode[dotstyle=o,dotsize=4pt](%g,%g){cnd%d}" %
                (cnd[0], cnd[1], icl))
            self.append("\\nput[labelsep=3pt]{270}{cnd%d}{\\texttt{%d}}" %
                (icl, icl))
        self.append("")
        self._has_cell_centers = True


def main():
    blk = create_sample_block()

    cmds = PsPicture(blk=blk, unit="2.5cm",
                     x0=-0.2, y0=-0.2, x1=3.2, y1=2.2)
    cmds.add_header()
    cmds.add_nodes(label=True)
    cmds.add_faces()
    cmds.add_cell_centers()
    cmds.add_footer()

    if not os.path.exists("schematic"):
        os.makedirs("schematic")
    with open(os.path.join("schematic", "mesh_2d.tex"), "w") as fobj:
        fobj.write("\n".join(cmds))

    if not os.path.exists("tmp"):
        os.makedirs("tmp")
    os.system("./pstake.py -q %s %s" % (
        os.path.join("schematic", "mesh_2d.tex"),
        os.path.join("tmp", "mesh_2d.eps")
    ))

if __name__ == '__main__':
    main()

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 tw=79:
