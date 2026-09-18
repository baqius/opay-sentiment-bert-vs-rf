
# import
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

wnl = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocessor(text):
    """Preprocessing used to train the RandomForest / TF-IDF model.
    Must stay IDENTICAL to the version used during training."""
    text = text.lower()
    text = re.sub(pattern='[^a-zA-Z0-9]', repl=' ', string=text)
    text = text.split()
    filtered_words = [word for word in text if word not in stop_words]
    lemmatized_words = [wnl.lemmatize(word) for word in filtered_words]
    
# import
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

wnl = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocessor(text):
    """Preprocessing used to train the RandomForest / TF-IDF model.
    Must stay IDENTICAL to the version used during training."""
    text = text.lower()
    text = re.sub(pattern='[^a-zA-Z0-9]', repl=' ', string=text)
    text = text.split()
    filtered_words = [word for word in text if word not in stop_words]
    lemmatized_words = [wnl.lemmatize(word) for word in filtered_words]
>>>>>>> 5ee785258d22274c711f58c73ad80a8030e984e5
    return ' '.join(lemmatized_words)
