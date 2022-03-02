import copy

import matplotlib.pyplot as plt
import napari
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSizePolicy, QSlider, QLabel
from magicgui import magicgui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pathlib import Path

from napari._qt.qthreading import thread_worker
from scipy import ndimage
import cc3d

import utils
from dock import Datamanager


def launch_viewers(original, base, raw, r_path, model_type, checkbox):
    global slicer
    global z_pos
    global view1
    global layer
    global images_original
    global base_label
    images_original = original
    base_label = base
    try:
        del view1
        del layer
    except NameError:
        pass
    view1 = napari.view_image(images_original, contrast_limits=[0, 255])
    view1.add_labels(base_label, name='base')
    if raw is not None:
        view1.add_image(ndimage.gaussian_filter(raw, sigma=3), colormap='magenta', name='low_confident',
                        blending='additive')
    else:
        pass

    def label_and_sort(base_label):
        labeled = ndimage.label(base_label, structure=np.ones((3, 3, 3)))[0]

        mks, nums = np.unique(labeled, return_counts=True)

        idx_list = list(np.argsort(nums[1:]))
        nums = np.sort(nums[1:])
        labeled_sorted = np.zeros_like(labeled)
        for i, idx in enumerate(idx_list):
            labeled_sorted = np.where(labeled == mks[1:][idx], i+1, labeled_sorted)
        return labeled_sorted, nums

    def label_ct(labeled_array, nums, value):
        labeled_temp = copy.copy(labeled_array)
        idx = np.abs(nums - value).argmin()
        labeled_temp = np.where((labeled_temp < idx) & (labeled_temp != 0), 255, 0)
        return labeled_temp

    def show_so_layer(args):
        labeled_c, labeled_sorted, nums = args
        so_layer = view1.add_image(labeled_c, colormap='cyan', name='small_object', blending='additive')

        object_slider = QSlider(Qt.Horizontal)
        object_slider.setMinimum(0)
        object_slider.setMaximum(100)
        object_slider.setSingleStep(10)
        object_slider.setValue(10)

        object_slider.valueChanged[int].connect(lambda value=object_slider: calc_object_callback(so_layer, value,
                                                                                                 labeled_sorted, nums))

        lbl = QLabel('object size')

        slider_widget = utils.combine_blocks(lbl, object_slider)

        view1.window.add_dock_widget(slider_widget, name='object_size_slider', area='left')

        def calc_object_callback(t_layer, value, labeled_array, nums):
            t_layer.data = label_ct(labeled_array, nums, value)

    @thread_worker(connect={"returned": show_so_layer})
    def create_label():
        labeled_sorted, nums = label_and_sort(base_label)
        labeled_c = label_ct(labeled_sorted, nums, 10)
        return labeled_c, labeled_sorted, nums

    worker = create_label()

    layer = view1.layers[0]
    layer1 = view1.layers[1]

    @magicgui(dirname={"mode": "d"})
    def dirpicker(dirname=Path(r_path)):
        """Take a filename and do something with it."""
        print("The filename is:", dirname)
        return dirname

    gui = dirpicker.Gui(show=True)
    view1.window.add_dock_widget(gui)

    @magicgui(call_button="save")
    def saver():
        out_dir = gui.dirname
        print("The directory is:", out_dir)
        return utils.save_masks(layer1.data, out_dir)

    gui2 = saver.Gui(show=True)
    view1.window.add_dock_widget(gui2, area='bottom')

    dmg = Datamanager()
    dmg.prepare(r_path, model_type, checkbox)
    view1.window.add_dock_widget(dmg, area='left')

    def update_button(axis_event):
        axis = axis_event.axis
        if axis != 0:
            return
        slice_num = axis_event.value
        dmg.update(slice_num)

    view1.dims.events.axis.connect(update_button)

    # draw canvas

    with plt.style.context('dark_background'):
        canvas = FigureCanvas(Figure(figsize=(3, 15)))

        xy_axes = canvas.figure.add_subplot(3, 1, 1)

        xy_axes.imshow(np.zeros((100, 100), np.uint8))
        xy_axes.scatter(50, 50, s=10, c='red', alpha=0.15)
        xy_axes.set_xlabel('x axis')
        xy_axes.set_ylabel('y axis')
        yz_axes = canvas.figure.add_subplot(3, 1, 2)
        yz_axes.imshow(np.zeros((100, 100), np.uint8))
        yz_axes.scatter(50, 50, s=10, c='red', alpha=0.15)
        yz_axes.set_xlabel('y axis')
        yz_axes.set_ylabel('z axis')
        zx_axes = canvas.figure.add_subplot(3, 1, 3)
        zx_axes.imshow(np.zeros((100, 100), np.uint8))
        zx_axes.scatter(50, 50, s=10, c='red', alpha=0.15)
        zx_axes.set_xlabel('x axis')
        zx_axes.set_ylabel('z axis')

        # canvas.figure.tight_layout()
        canvas.figure.subplots_adjust(left=0, bottom=0.1, right=1, top=0.95, wspace=0, hspace=0.4)

    canvas.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)

    view1.window.add_dock_widget(canvas, area='right')

    @layer.mouse_drag_callbacks.append
    def update_canvas_canvas(layer, event):
        if 'shift' in event.modifiers:
            try:
                m_point = np.round(layer.coordinates).astype(int)
                print(m_point)
                crop_big = crop_img([m_point[0], m_point[1], m_point[2]], layer)
                xy_axes.imshow(crop_big[50], 'gray')
                yz_axes.imshow(crop_big.transpose(1, 0, 2)[50], 'gray')
                zx_axes.imshow(crop_big.transpose(2, 0, 1)[50], 'gray')
                canvas.draw_idle()
            except Exception as e:
                print(e)

    def crop_img(points, layer):
        min_vals = [x-50 for x in points]
        max_vals = [x+50 for x in points]
        yohaku_minus = [n if n < 0 else 0 for n in min_vals]
        yohaku_plus = [x - layer.data.shape[i] if layer.data.shape[i] < x else 0 for i, x in
                       enumerate(max_vals)]
        crop_slice = tuple(slice(np.maximum(0, n), x) for n, x in zip(min_vals, max_vals))
        crop_temp = layer.data[crop_slice].persist().compute()
        cropped_img = np.zeros((100, 100, 100), np.uint8)
        cropped_img[-yohaku_minus[0]:100 - yohaku_plus[0], -yohaku_minus[1]:100 - yohaku_plus[1]
        , -yohaku_minus[2]:100 - yohaku_plus[2]] = crop_temp
        return cropped_img


def launch_selector(original, label, select, mod_path, select_path):

    global view1
    global layer1
    global layer2
    global images_original
    global base_label
    global only_label

    images_original = original
    base_label = label
    only_label = select

    # remove duplicate area from base_label 
    base_label = np.where(only_label>0, 0, base_label)

    try:
        del view1
        del layer1
        del layer2
    except NameError:
        pass
    view1 = napari.view_image(images_original, contrast_limits=[0, 255])
    view1.add_labels(base_label, name='base', color={1:'red', 2:'blue', 3:'green'})
    view1.add_labels(only_label, name="selected_objects", color={1:'blue', 2:'green', 3:'red'})

    # culc label
    labels_imgs = cc3d.connected_components((label == 1).astype(int), connectivity = 6)
    layer1 = view1.layers[1]
    layer2 = view1.layers[2]

    def select_target():
        global base_label
        global only_label
    
        @layer1.bind_key('q', overwrite = True)
        def select(layer):
            global base_label
            global only_label
            q_point = np.round(layer1.coordinates).astype(int)
            print("select_label : ", labels_imgs[q_point[0], q_point[1], q_point[2]], "select_point : ", q_point)
            if labels_imgs[q_point[0], q_point[1], q_point[2]]:
                target_label = labels_imgs[q_point[0], q_point[1], q_point[2]]
                only_label = np.where(labels_imgs == target_label, 1, only_label)
                base_label = np.where(labels_imgs == target_label, 0, base_label)
                view1.layers[1].data = base_label
                view1.layers[2].data = only_label
                print('Selected!')
            return True

    @layer1.mouse_drag_callbacks.append
    def main(layer, event):
        select_target()

    def deselect_target():
        global base_label
        global only_label

        @layer2.bind_key('w', overwrite = True)
        def deselect(layer):
            global base_label
            global only_label
            w_point = np.round(layer2.coordinates).astype(int)
            print("select_label : ", labels_imgs[w_point[0], w_point[1], w_point[2]], "select_point : ", w_point)
            if labels_imgs[w_point[0], w_point[1], w_point[2]]:
                target_label = labels_imgs[w_point[0], w_point[1], w_point[2]]
                base_label = np.where(labels_imgs == target_label, 1, base_label)
                only_label = np.where(labels_imgs == target_label, 0, only_label)
                view1.layers[1].data = base_label
                view1.layers[2].data = only_label
                print('Deselected!')
            return True

    @layer2.mouse_drag_callbacks.append
    def main(layer, event):
        deselect_target()

    @magicgui(dirname={"mode": "d"})
    def mod_dirpicker(dirname=Path(mod_path)):
        """Take a filename and do something with it."""
        print("The filename is:", dirname)
        return dirname

    gui1 = mod_dirpicker.Gui(show=True)
    view1.window.add_dock_widget(gui1, area='bottom')

    @magicgui(dirname={"mode": "d"})
    def select_dirpicker(dirname=Path(select_path)):
        """Take a filename and do something with it."""
        print("The filename is:", dirname)
        return dirname

    gui3 = select_dirpicker.Gui(show=True)
    view1.window.add_dock_widget(gui3, area='bottom')

    @magicgui(call_button="save")
    def saver():
        print("The directory is:", select_path)
        return utils.save_masks(layer1.data, mod_path), utils.save_masks(layer2.data, select_path)

    gui4 = saver.Gui(show=True)
    view1.window.add_dock_widget(gui4, area='bottom')