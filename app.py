
import math
from lxml import etree
from predict import PalletePredictor
from skimage import color


from flask import Flask, render_template, request, redirect, session
import matplotlib

matplotlib.use('Agg')
UPLOAD_FOLDER = '/Desktop/test'
ALLOWED_EXTENSIONS = set(['svg'])
colour_palette = ['#e68558', '#efb467', '#faf096', '#c4dfe3', '#99bcc2']
list_of_id = {'body': '#ffffff', 'background': 'None', 'button-primary': 'None', 'header': 'None', 'footer': 'None',
              'icon': '#e2dfdf', 'image': '#e2dfdf', 'table': '#e2dfdf', 'navbar': '#e2dfdf',
              'button-secondary': '#e2dfdf', 'card': '#f5f2f2', 'search': '#ffffff'}
list_of_id2 = {'headline-h1': 'None', 'text': 'None'}
list_of_id3 = {'headline-h2': 'None'}
list_of_id4 = {'button-primary': 'None'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def contrast(rgb1, rgb2):
    return (luminance(rgb1) + 0.05) / (luminance(rgb2) + 0.05)


def luminance(rgb):
    r, g, b = [norm(c) for c in rgb]
    return r * 0.2126 + g * 0.7152 + b * 0.0722


def norm(v):
    v /= 255
    if v <= 0.03928:
        return v / 12.92
    else:
        return math.pow((v + 0.055) / 1.055, 2.4)


def hex_to_rgb(hexcolour):
    i = hexcolour.lstrip('#')
    c = int(i, 16)
    r = (c&0xff0000) >> 16
    g = (c&0xff00) >> 8
    b = c&0xff
    return [r,g,b]


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)

def coloursort(pal, index):

    i=0
    maxcon = 0.001
    maxind = 0
    print(pal[index])
    print(pal[0])
    i+=1
    while i<5:
        if i != index:
            a = abs(color.deltaE_ciede2000(pal[index], pal[i])[0][0])
            if a>maxcon:
                maxcon = a
                maxind = i
        i+=1
    i=0
    maxcon = 0.001
    maxind2 = 0
    while i<5:
        if i != index and i != maxind:
            a = abs(color.deltaE_ciede2000(pal[i], pal[maxind])[0][0])
            if a>maxcon:
                maxcon = a
                maxind2 = i
        i+=1
    inds = [index, maxind, maxind2]
    return inds


def coloring(palette, tree):
    etree.tostring(tree)
    root = tree.getroot()
    print(root.tag)
    child = root[0]
    list_of_id_edited = list_of_id
    list_of_id_edited['background'] = palette[0]
    list_of_id_edited['button-primary'] = palette[4]
    list_of_id_edited['header'] = palette[1]
    list_of_id_edited['footer'] = palette[1] 
    for element in root.iter():
        print(element.get("id"))
        print('been here')
        if element.get("id") is not None:
            for i in list_of_id.keys():
                if element.get("id").find(i)==0:
                    print('here too')
                    if element.get("fill") is not None:
                        print('right here too')
                        attributes = element.attrib
                        attributes["fill"] = list_of_id_edited[i]
                    else:
                        if element[0].get("fill") is not None:
                            print('and there too')   
                            attributes = element[0].attrib
                            attributes["fill"] = list_of_id_edited[i]
    for element in root.iter():
        print((element.get("id")))
        print('bam')
        if element.get("id") is not None:
            for i in list_of_id2.keys():
                if element.get("id").find(i)==0:
                    print('babam')
                    midroot = element.getparent()
                    attributes = element.attrib
                    flag = False
                    for i in list_of_id.keys():
                        if midroot[0].get("id").find(i)==0:
                            flag = True
                            print('spray')
                            if contrast([255, 255, 255], hex_to_rgb(midroot[0].get("fill")))<4.5:
                                attributes["fill"] = "#000000"
                            else:
                                attributes["fill"] = "#ffffff"
                    if not flag:
                        attributes["fill"] = "#000000"
            for i in list_of_id3.keys():
                if element.get("id").find(i)==0:
                    print('test')
                    midroot = element.getparent()
                    attributes = element.attrib
                    for i in list_of_id.keys():
                        if midroot[0].get("id").find(i)==0:
                            flag = True
                            print('spray')
                            if contrast([255, 255, 255], hex_to_rgb(midroot[0].get("fill")))<4.5:
                                attributes["fill"] = "#353535"
                            else:
                                attributes["fill"] = "#ffffff"
                    if not flag:
                        attributes["fill"] = "#353535"
    for element in root.iter():
        if element.get("id") is not None:
            for i in list_of_id4.keys():
                if element.get("id").find(i) == 0:
                    element.set("rx", "20")

    print(etree.tostring(root, pretty_print=True))
    return root


@app.route('/choose_colour', methods=['POST', 'GET'])
def choose_colour():
    temp_palette1 = session['chosen_palette']
    temp_hex1 = []
    i = 0
    j = 0
    while j < 5:
        temp_rgb = [int(round(temp_palette1[i] * 255)), int(round(temp_palette1[i + 1] * 255)),
                    int(round(temp_palette1[i+2] * 255))]
        temp_hex1.append(rgb_to_hex(temp_rgb))
        j += 1
        i += 3
    print(temp_hex1)
    if request.method == "POST":
        i=1
        while i<=5:
            a = "col{}"
            if request.form.get(a.format(i)) is not None:
                break
            i+=1
        print(i)
        temp_list = session['chosen_palette']
        print(temp_list)
        stemplist = []
        j = 0
        for k in range(5):
            stemplist.append([[[temp_list[j], temp_list[j+1], temp_list[j+2]]]])
            j+=3
        labtemp = []
        for k in range(5):
            labtemp.append(color.rgb2lab(stemplist[k]))
        colorinds = coloursort(labtemp, i-1)
        print(colorinds)

        if colorinds[0] != 0:
            for i in range(3):
                temp_list[i], temp_list[colorinds[0]*3+i] = temp_list[colorinds[0]*3+i], temp_list[i]
        print(temp_list)
        if colorinds[1] == 0:
            colorinds[1] = colorinds[0]
        if colorinds[2] == 0:
            colorinds[2] = colorinds[0]
        if colorinds[1] != 4:
            for i in range(3):
                temp_list[12+i], temp_list[colorinds[1]*3+i] = temp_list[colorinds[1]*3+i], temp_list[12+i]
        print(temp_list)
        if colorinds[2] == 4:
            colorinds[2] = colorinds[1]
        print(colorinds[2])
        if colorinds[2] != 1:
            for i in range(3):
                temp_list[3+i], temp_list[colorinds[2]*3+i] = temp_list[colorinds[2]*3+i], temp_list[3+i]
        print(temp_list)
        temp_hex2 = []
        i = 0
        j = 0
        while j < 5:
            temp_rgb = [int(round(temp_list[i] * 255)), int(round(temp_list[i + 1] * 255)),
                        int(round(temp_list[i+2] * 255))]
            temp_hex2.append(rgb_to_hex(temp_rgb))
            j += 1
            i += 3
        print(temp_hex2)
        session['colour_palette_sorted'] = temp_list
        tree = etree.parse('coloured_svg.svg')
        str = etree.tostring(tree)
        print(tree)
        print(str)
        str = etree.tostring(coloring(temp_hex2, tree), pretty_print=True)
        temp_hex2[1], temp_hex2[2] = temp_hex2[2], temp_hex2[1]
        str2 = etree.tostring(coloring(temp_hex2, tree), pretty_print=True)
        with open('finalsvg1.svg', 'wb') as file2:
                file2.write(str)
        with open('finalsvg2.svg', 'wb') as file2:
                file2.write(str2)
        return redirect('/generated')
    return render_template("choose_colour.html", colours=temp_hex1)


@app.route('/generated')
def generated():
    return render_template("generated.html")


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        file = request.files['file']
        if file and allowed_file(file.filename):
            tree = etree.parse(file)
            str = etree.tostring(tree)
            with open('coloured_svg.svg', 'wb') as file:
                file.write(str)
                file.close()
            return redirect('/input_text')
        else:
            return redirect('/')
    else:
        return render_template("index.html")


@app.route('/uploaded_file')
def uploaded_file():
    return render_template("file_uploaded.html")


@app.route('/input_text', methods=['POST', 'GET'])
def input_text():
    if request.method == "POST":
        textInput = request.form['text_input']
        print(textInput)
        predictor = PalletePredictor()
        new_palette1 = predictor.get_pallete(textInput)

        print('первая')
        new_palette1_rgb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        i = 0
        while i < 5:
            j = 0
            while j < 3:
                new_palette1_rgb[i * 3 + j] = new_palette1[i][j]
                j += 1
            i += 1
        print(new_palette1_rgb)
        k=0
        print('вторая')
        new_palette2_rgb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        while i < 10:
            j = 0
            while j < 3:
                new_palette2_rgb[k * 3 + j] = new_palette1[i][j]
                j += 1
            i += 1
            k += 1
        print(new_palette2_rgb)

        print('третья')
        new_palette3_rgb = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        k=0
        while i<15:
            j=0
            while j<3:
                new_palette3_rgb[k*3+j]=new_palette1[i][j]
                j+=1
            i+=1
            k+=1
        print(new_palette3_rgb)

        print('четвертая')
        new_palette4_rgb = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        k=0
        while i<20:
            j=0
            while j<3:
                new_palette4_rgb[k*3+j]=new_palette1[i][j]
                j+=1
            i+=1
            k+=1
        print(new_palette4_rgb)

        print('пятая')
        new_palette5_rgb = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        k=0
        while i<25:
            j=0
            while j<3:
                new_palette5_rgb[k*3+j]=new_palette1[i][j]
                j+=1
            i+=1
            k+=1
        print(new_palette5_rgb)

        # print(new_palette1[0][0]*255)

        # print(int(test*255))

        session['palette1'] = new_palette1_rgb
        session['palette2'] = new_palette2_rgb
        session['palette3'] = new_palette3_rgb
        session['palette4'] = new_palette4_rgb
        session['palette5'] = new_palette5_rgb
        return redirect('/generated_palettes')

    return render_template("input_text.html")

@app.route('/generated_palettes', methods=['POST', 'GET'])
def generated_palettes():
    if request.method == "POST":
        i = 1
        while i <= 5:
            a = "col{}"
            if request.form.get(a.format(i)) is not None:
                break
            i += 1
        print(i)
        print('palette{}'.format(i))
        session['chosen_palette'] = session['palette{}'.format(i)]
        return redirect('/choose_colour')
    temp_palette1 = session['palette1']
    temp_palette2 = session['palette2']
    temp_palette3 = session['palette3']
    temp_palette4 = session['palette4']
    temp_palette5 = session['palette5']

    temp_hex1 = []
    i=14
    j=0
    while j<5:
        temp_rgb = [int(round(temp_palette1[i-2]*255)), int(round(temp_palette1[i-1]*255)), int(round(temp_palette1[i]*255))]
        temp_hex1.append(rgb_to_hex(temp_rgb))
        j+=1
        i-=3
    print(temp_hex1)
    temp_hex2 = []
    i = 14
    j = 0
    while j<5:
        temp_rgb = [int(round(temp_palette2[i-2]*255)), int(round(temp_palette2[i-1]*255)), int(round(temp_palette2[i]*255))]
        temp_hex2.append(rgb_to_hex(temp_rgb))
        j+=1
        i-=3
    print(temp_hex2)
    temp_hex3 = []
    i = 14
    j = 0
    while j<5:
        temp_rgb = [int(round(temp_palette3[i-2]*255)), int(round(temp_palette3[i-1]*255)), int(round(temp_palette3[i]*255))]
        temp_hex3.append(rgb_to_hex(temp_rgb))
        j+=1
        i-=3
    print(temp_hex3)
    temp_hex4 = []
    i = 14
    j = 0
    while j<5:
        temp_rgb = [int(round(temp_palette4[i-2]*255)), int(round(temp_palette4[i-1]*255)), int(round(temp_palette4[i]*255))]
        temp_hex4.append(rgb_to_hex(temp_rgb))
        j+=1
        i-=3
    print(temp_hex4)
    temp_hex5 = []
    i = 14
    j = 0
    while j<5:
        temp_rgb = [int(round(temp_palette5[i-2]*255)), int(round(temp_palette5[i-1]*255)), int(round(temp_palette5[i]*255))]
        temp_hex5.append(rgb_to_hex(temp_rgb))
        j+=1
        i-=3
    print(temp_hex5)
    return render_template("generated_palettes.html", palette1=temp_hex1, palette2=temp_hex2, palette3=temp_hex3,
                           palette4=temp_hex4, palette5=temp_hex5)



if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)
