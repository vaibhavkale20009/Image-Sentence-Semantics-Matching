from flask import Flask, render_template, redirect, request
from predict import load_model, predict_caption ,cosine_score,get_seman,predictc
from PIL import Image
from flask_ngrok import run_with_ngrok


app = Flask(__name__) 
run_with_ngrok(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
@app.route('/')
def hello():
    return render_template("index.html")





ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):

    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods = ['POST'])
def submit_data():
    if request.method == 'POST':
        f = request.files['userfile']
        if not allowed_file(f.filename):
            return "ERROR : Please upload JPEG, PNG or JPG format"
        
        path = "/content/sample_data/{}".format(f.filename)
        f.save(path)
       
        captions = predict_caption(path)
        tmpcaption = predictc(path)
        semantics= get_seman(tmpcaption)
        result_dict ={
            'image': path,
            'captions' : captions,
            'semantics' : semantics
        }
         
    
    return render_template("index.html", captions= result_dict) 




if __name__ == "__main__":
    load_model()
    app.run()