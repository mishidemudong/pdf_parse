#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:44:17 2020

@author: liang
"""

import multiprocessing
import argparse
import pdfplumber
import os
from tqdm import tqdm
from pdfminer.layout import LTChar, LTLine
import re
from collections import Counter
import pdf2image
import numpy as np
from PIL import Image

def bbox_string(box, width, length):
    return (
        str(int(1000 * (box[0] / width)))
        + " "
        + str(int(1000 * (box[1] / length)))
        + " "
        + str(int(1000 * (box[2] / width)))
        + " "
        + str(int(1000 * (box[3] / length)))
    )


def actual_bbox_string(box, width, length):
    return (
        str(box[0])
        + " "
        + str(box[1])
        + " "
        + str(box[2])
        + " "
        + str(box[3])
        + "\t"
        + str(width)
        + " "
        + str(length)
    )

def within_bbox(bbox_bound, bbox_in):
    assert bbox_bound[0] <= bbox_bound[2]
    assert bbox_bound[1] <= bbox_bound[3]
    assert bbox_in[0] <= bbox_in[2]
    assert bbox_in[1] <= bbox_in[3]

    x_left = max(bbox_bound[0], bbox_in[0])
    y_top = max(bbox_bound[1], bbox_in[1])
    x_right = min(bbox_bound[2], bbox_in[2])
    y_bottom = min(bbox_bound[3], bbox_in[3])

    if x_right < x_left or y_bottom < y_top:
        return False

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox_in_area = (bbox_in[2] - bbox_in[0]) * (bbox_in[3] - bbox_in[1])

    if bbox_in_area == 0:
        return False

    iou = intersection_area / float(bbox_in_area)

    return iou > 0.95


def parsepdf(pdf_file, data_dir, output_dir):


    try:
        pdf_images = pdf2image.convert_from_path(os.path.join(data_dir, pdf_file))
    except:
        return

    page_tokens = []
    try:
        pdf = pdfplumber.open(os.path.join(data_dir, pdf_file))
    except:
        return
    
    for page_id in tqdm(range(len(pdf.pages))):
        tokens = []
    
        this_page = pdf.pages[page_id]
        anno_img = np.ones([int(this_page.width), int(this_page.height)] + [3], dtype=np.uint8) * 255
    
        words = this_page.extract_words(x_tolerance=1.5)
    
        lines = []
        for obj in this_page.layout._objs:
            if not isinstance(obj, LTLine):
                continue
            lines.append(obj)
    
        for word in words:
            word_bbox = (float(word['x0']), float(word['top']), float(word['x1']), float(word['bottom']))
            objs = []
            for obj in this_page.layout._objs:
                if not isinstance(obj, LTChar):
                    continue
                obj_bbox = (obj.bbox[0], float(this_page.height) - obj.bbox[3],
                            obj.bbox[2], float(this_page.height) - obj.bbox[1])
                if within_bbox(word_bbox, obj_bbox):
                    objs.append(obj)
            fontname = []
            for obj in objs:
                fontname.append(obj.fontname)
            if len(fontname) != 0:
                c = Counter(fontname)
                fontname, _ = c.most_common(1)[0]
            else:
                fontname = 'default'
    
            # format word_bbox
            width = int(this_page.width)
            height = int(this_page.height)
            f_x0 = word_bbox[0]
            f_y0 = word_bbox[1]
            f_x1 = word_bbox[2]
            f_y1 = word_bbox[3]
            word_bbox = tuple([f_x0, f_y0, f_x1, f_y1, width, height])
    
            # plot annotation
            x0, y0, x1, y1,_, _ = word_bbox
            x0, y0, x1, y1 = int(x0 * width / 1000), int(y0 * height / 1000), int(x1 * width / 1000), int(
                y1 * height / 1000)
            anno_color = [0, 0, 0]
            for x in range(x0, x1):
                for y in range(y0, y1):
                    anno_img[x, y] = anno_color

            word_bbox = tuple([str(t) for t in word_bbox])
            word_text = re.sub(r"\s+", "", word['text'])
            tokens.append((word_text,) + word_bbox + (fontname,))
    
        for figure in this_page.figures:
            figure_bbox = (float(figure['x0']), float(figure['top']), float(figure['x1']), float(figure['bottom']))
    
            # format word_bbox
            width = int(this_page.width)
            height = int(this_page.height)
            f_x0 = figure_bbox[0]
            f_y0 = figure_bbox[1]
            f_x1 = figure_bbox[2] 
            f_y1 = figure_bbox[3]
            figure_bbox = tuple([f_x0, f_y0, f_x1, f_y1, width, height])
    
            # plot annotation
            x0, y0, x1, y1,_, _ = figure_bbox
            x0, y0, x1, y1 = int(x0 * width / 1000), int(y0 * height / 1000), int(x1 * width / 1000), int(
                y1 * height / 1000)           
            anno_color = [0, 0, 0]
            for x in range(x0, x1):
                for y in range(y0, y1):
                    anno_img[x, y] = anno_color
                    
            figure_bbox = tuple([str(t) for t in figure_bbox])
            word_text = '##LTFigure##'
            fontname = 'default'
            tokens.append((word_text,) + figure_bbox + (fontname,))
    
        for line in this_page.lines:
            line_bbox = (float(line['x0']), float(line['top']), float(line['x1']), float(line['bottom']))
            # format word_bbox
            width = int(this_page.width)
            height = int(this_page.height)
            f_x0 = line_bbox[0] 
            f_y0 = line_bbox[1]
            f_x1 = line_bbox[2] 
            f_y1 = line_bbox[3]
            line_bbox = tuple([f_x0, f_y0, f_x1, f_y1, width, height])
    
            # plot annotation
            x0, y0, x1, y1,_, _ = line_bbox
            x0, y0, x1, y1 = int(x0 * width / 1000), int(y0 * height / 1000), int(x1 * width / 1000), int(
                y1 * height / 1000)
            anno_color = [0, 0, 0]
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    anno_img[x, y] = anno_color
    
            line_bbox = tuple([str(t) for t in line_bbox])
            word_text = '##LTLine##'
            fontname = 'default'
            tokens.append((word_text,) + line_bbox + (fontname, ))
    
        anno_img = np.swapaxes(anno_img, 0, 1)
        anno_img = Image.fromarray(anno_img, mode='RGB')
        page_tokens.append((page_id, tokens, anno_img))
        
        pdf_images[page_id].save(
            os.path.join(output_dir, pdf_file.replace('.pdf', '') + '_{}_ori.jpg'.format(str(page_id))))
        anno_img.save(
            os.path.join(output_dir, pdf_file.replace('.pdf', '') + '_{}_ann.jpg'.format(str(page_id))))
        
        with open(os.path.join(output_dir, pdf_file.replace('.pdf', '') + '_{}.txt'.format(str(page_id))),
                    'w',
                    encoding='utf8') as fp:
            for token in tokens:
                fp.write('\t'.join(token) + '\n')
                
        return page_tokens
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default="./inputdata",
        type=str,
        # required=True,
        help="The input data dir. Should contain the pdf files.",
    )
    parser.add_argument(
        "--output_dir",
        default="./outputdata",
        type=str,
        # required=True,
        help="The output directory where the output data will be written.",
    )
    args = parser.parse_args()

    pdf_files = list(os.listdir(args.data_dir))#[:10]
    pdf_files = [t for t in pdf_files if t.endswith('.pdf')]
    
    alldata = []
    # # pool = multiprocessing.Pool(processes=1)
    for pdf_file in tqdm(pdf_files):
        # pool.apply_async(worker, (pdf_file, args.data_dir, args.output_dir))
        # alldata.append(parsepdf(pdf_file, args.data_dir, args.output_dir))
        
        pdf_images = pdf2image.convert_from_path(os.path.join( args.data_dir, pdf_file))

    
        page_tokens = []
        pdf = pdfplumber.open(os.path.join(args.data_dir, pdf_file))

        
        for page_id in tqdm(range(len(pdf.pages))):
            # if page_id != 7:
            #     continue
            tokens = []
        
            this_page = pdf.pages[page_id]
            anno_img = np.ones([int(this_page.width), int(this_page.height)] + [3], dtype=np.uint8) * 255
        
            words = this_page.extract_words(x_tolerance=1.5)
        
            lines = []
            for obj in this_page.layout._objs:
                if not isinstance(obj, LTLine):
                    continue
                lines.append(obj)
        
            for word in words:
                word_bbox = (float(word['x0']), float(word['top']), float(word['x1']), float(word['bottom']))
                objs = []
                for obj in this_page.layout._objs:
                    if not isinstance(obj, LTChar):
                        continue
                    obj_bbox = (obj.bbox[0], float(this_page.height) - obj.bbox[3],
                                obj.bbox[2], float(this_page.height) - obj.bbox[1])
                    if within_bbox(word_bbox, obj_bbox):
                        objs.append(obj)
                fontname = []
                for obj in objs:
                    fontname.append(obj.fontname)
                if len(fontname) != 0:
                    c = Counter(fontname)
                    fontname, _ = c.most_common(1)[0]
                else:
                    fontname = 'default'
        
                # format word_bbox
                width = int(this_page.width)
                height = int(this_page.height)
                f_x0 = word_bbox[0]
                f_y0 = word_bbox[1]
                f_x1 = word_bbox[2]
                f_y1 = word_bbox[3]
                word_bbox = tuple([f_x0, f_y0, f_x1, f_y1, width, height])
        
                # plot annotation
                x0, y0, x1, y1,_, _ = word_bbox
                x0, y0, x1, y1 = int(x0 * width / 1000), int(y0 * height / 1000), int(x1 * width / 1000), int(
                    y1 * height / 1000)
                anno_color = [0, 0, 0]
                for x in range(x0, x1):
                    for y in range(y0, y1):
                        anno_img[x, y] = anno_color
    
                word_bbox = tuple([str(t) for t in word_bbox])
                word_text = re.sub(r"\s+", "", word['text'])
                tokens.append((word_text,) + word_bbox + (fontname,))
        
            for figure in this_page.figures:
                figure_bbox = (float(figure['x0']), float(figure['top']), float(figure['x1']), float(figure['bottom']))
        
                # format word_bbox
                width = int(this_page.width)
                height = int(this_page.height)
                f_x0 = figure_bbox[0]
                f_y0 = figure_bbox[1]
                f_x1 = figure_bbox[2] 
                f_y1 = figure_bbox[3]
                figure_bbox = tuple([f_x0, f_y0, f_x1, f_y1, width, height])
        
                # plot annotation
                x0, y0, x1, y1,_, _ = figure_bbox
                x0, y0, x1, y1 = int(x0 * width / 1000), int(y0 * height / 1000), int(x1 * width / 1000), int(
                    y1 * height / 1000)           
                anno_color = [0, 0, 0]
                for x in range(x0, x1):
                    for y in range(y0, y1):
                        anno_img[x, y] = anno_color
                        
                figure_bbox = tuple([str(t) for t in figure_bbox])
                word_text = '##LTFigure##'
                fontname = 'default'
                tokens.append((word_text,) + figure_bbox + (fontname,))
        
            for line in this_page.lines:
                line_bbox = (float(line['x0']), float(line['top']), float(line['x1']), float(line['bottom']))
                # format word_bbox
                width = int(this_page.width)
                height = int(this_page.height)
                f_x0 = line_bbox[0] 
                f_y0 = line_bbox[1]
                f_x1 = line_bbox[2] 
                f_y1 = line_bbox[3]
                line_bbox = tuple([f_x0, f_y0, f_x1, f_y1, width, height])
        
                # plot annotation
                x0, y0, x1, y1,_, _ = line_bbox
                x0, y0, x1, y1 = int(x0 * width / 1000), int(y0 * height / 1000), int(x1 * width / 1000), int(
                    y1 * height / 1000)
                anno_color = [0, 0, 0]
                for x in range(x0, x1 + 1):
                    for y in range(y0, y1 + 1):
                        anno_img[x, y] = anno_color
        
                line_bbox = tuple([str(t) for t in line_bbox])
                word_text = '##LTLine##'
                fontname = 'default'
                tokens.append((word_text,) + line_bbox + (fontname, ))
        
            anno_img = np.swapaxes(anno_img, 0, 1)
            anno_img = Image.fromarray(anno_img, mode='RGB')
            page_tokens.append((page_id, tokens, anno_img))
            
            pdf_images[page_id].save(
                os.path.join(args.output_dir, pdf_file.replace('.pdf', '') + '_{}_ori.jpg'.format(str(page_id))))
            anno_img.save(
                os.path.join(args.output_dir, pdf_file.replace('.pdf', '') + '_{}_ann.jpg'.format(str(page_id))))
            
            with open(os.path.join(args.output_dir, pdf_file.replace('.pdf', '') + '_{}.txt'.format(str(page_id))),
                        'w',
                        encoding='utf8') as fp:
                for token in tokens:
                    fp.write('\t'.join(token) + '\n')

    # pool.close()
    # pool.join()
