from flask import Flask, render_template, redirect, request

import caption

# __name__ == __main__
app = Flask(__name__)


@app.route('/')
def hello():
	return render_template("index.html")


@app.route('/', methods= ['POST'])
def marks():
	if request.method == 'POST':

		f = request.files['userfile']
		path = "./static/{}".format(f.filename)# ./static/images.jpg
		f.save(path)

		caption_img = caption.predict_caption(path)
		
		result_dic = {
		'image' : path,
		'caption' : caption_img
		}

	return render_template("index.html", your_result =result_dic)

if __name__ == '__main__':
    app.run(debug = False, threaded = False)