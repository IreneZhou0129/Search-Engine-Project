from spider import vsm , boolean_retrieval , courses , vsm_query_processing
from flask import Flask, render_template, request, url_for,redirect, session , jsonify , json

app = Flask(__name__)


@app.route('/',methods=['POST','GET'])
def main():
    if request.method == 'POST':
        if request.form['search-button'] == 'search':
            
            return redirect(url_for('result'))
    else:
        return render_template('index.html')

@app.route('/result',methods=['POST'])
def result():
    query = request.form['search-bar']
    module_type = request.form['module']
    
    edit_dist = None
    formatter = lambda i: (courses[i],None)
    if(module_type == "boolean"):
        rank, query_expansion = boolean_retrieval(query,courses)
    else:
        rank,test_vsm , query_expansion = vsm_query_processing(query,vectorizer, matrix ,tf_idf)
    
    ranked_doc = [courses[i] for i in rank[:5]]
    
    return render_template('result.html', 
                    ranked_doc = ranked_doc[:5],
                    query = query)

@app.route('/document/<code>', methods=['GET','POST'])
def document(code):
    course = courses[int(code)]
    # dict_keys(['code', 'name', 'docID', 'description'])
    return render_template('one_doc.html',
                course=course)


if __name__ == '__main__':
    vectorizer , matrix, tf_idf = vsm(courses)
    app.run(debug=False)