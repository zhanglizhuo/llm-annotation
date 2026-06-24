import re
with open('../llm_annotation_paper_plos.tex', 'r') as f:
    content = f.read()

# match content between \section*{Introduction} or similar and the end
res = re.search(r'\\section\{Introduction\}(.*?)\\section\*\{Author contributions\}', content, re.DOTALL)
if res:
    body = res.group(1)
    
    # We need to drop pgfplots figures or replace them with incudegraphics
    body = re.sub(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', '[Figure placeholder - converting from TikZ]', body, flags=re.DOTALL)
    
    with open('access_submission.tex', 'r') as f2:
        template = f2.read()
    
    new_template = template.replace('% Please wait while I migrate the rest of the text safely.', body)
    
    with open('access_submission.tex', 'w') as f3:
        f3.write(new_template)
