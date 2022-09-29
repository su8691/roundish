import os
os.chdir("..")
import sys 
sys.path.append(os.getcwd())
import time
import xml.etree.ElementTree as ET
from deepsvg.svglib.geom import Point
from deepsvg.svglib.svg import SVG
from deepsvg.svglib.svg_path import SVGPath
from deepsvg.svglib.utils import to_gif
from deepsvg.svglib.svg_command import *

from deepsvg.difflib.tensor import SVGTensor
from deepsvg.difflib.utils import *
from deepsvg.difflib.loss import *

import torch.optim as optim
import IPython.display as ipd
from moviepy.editor import ImageClip, concatenate_videoclips, ipython_display
from cairosvg import svg2png


allFileList = os.listdir("docs/test")
#欲改作字型svg資料夾路徑

start = time.time()
total_start = start
path = 'docs/time_output.txt'
o = open(path, 'w')

for file in allFileList:

    tree = ET.parse("docs/test/" + file)
    ET.register_namespace('', "http://www.w3.org/2000/svg")
    root = tree.getroot()
    path = root.find('{http://www.w3.org/2000/svg}path')

    if 'd' in path.attrib:
        svg = SVG.load_svg("docs/test/" + file).normalize().zoom(0.9).canonicalize().simplify_heuristic()
        o.write(file)

        svg_target = SVGTensor.from_data(svg.to_tensor())
        p_target = svg_target.sample_points()
        #plot_points(p_target, show_color=True)

        shape = SVG.unit_circle().normalize().zoom(0.9).split(4) 
        svg_pred = SVGTensor.from_data(shape.to_tensor())

        p_pred = svg_pred.sample_points()
        #plot_points(p_pred, show_color=True)

        svg_pred.control1.requires_grad_(True)
        svg_pred.control2.requires_grad_(True)
        svg_pred.end_pos.requires_grad_(True)

        optimizer = optim.Adam([svg_pred.control1, svg_pred.control2, svg_pred.end_pos], lr=2)

        img_list = []

        for i in range(250):
            optimizer.zero_grad()

            p_pred = svg_pred.sample_points()
            l = svg_emd_loss(p_pred, p_target)
            l.backward()
            optimizer.step()
            
            if i % 249 == 0:
                #img = svg_pred.draw(with_points=True, do_display=False, return_png=True)
                svgtemp = SVG.from_tensor(svg_pred.data)
                svgtemp.save_svg("docs/result/" + file)

        # svg2png
        print('docs/result/' + file)
        f = open('docs/result/' + file ,encoding="utf-8")
        line = f.readline()
        svg2png(bytestring = line, write_to= 'docs/result/png/U+'+ file[2:-4] +'.png')
        f.close()

        end = time.time()
        o.write(" " + str(end - start) + "\n")
        start = end


o.write(str(end - total_start))    
o.close()