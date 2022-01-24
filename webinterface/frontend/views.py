from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from .Retrieval import search
from .Retrieval import initalize
from collections import Counter

# Create your views here.
def home(request):
    if request.method == 'POST' and request.FILES['filename']:
        image = request.FILES["filename"]
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        uploaded_file_url = fs.url(filename)
        img_paths, plants = search(filename)
        results = zip(img_paths, plants)

        occurence_count = Counter(plants)

        freq_label = occurence_count.most_common(1)[0][0]

        return render(request, 'home.html', {
            'image_url': uploaded_file_url,
            'result_list': results,
            'predicted_label': freq_label
        })
    initalize()
    return render(request, 'home.html')