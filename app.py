import streamlit as st
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords as nltk_stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Sentiment Analysis MyXL",
    page_icon="📱"
)


# ==========================================
# DOWNLOAD STOPWORDS NLTK
# ==========================================
nltk.download('stopwords', quiet=True)


# ==========================================
# LOAD MODEL DAN TF-IDF
# ==========================================
model = joblib.load("svm_calibrated_model.pkl")
tfidf = joblib.load("tfidf.pkl")


# ==========================================
# SLANG DICTIONARY
# ==========================================
slang_dict = {
    # PARTIKEL & KONJUNGSI
    'yg': 'yang',
    'tp': 'tapi',
    'tpi': 'tapi',
    'jg': 'juga',
    'dg': 'dengan',
    'dgn': 'dengan',
    'sm': 'sama',
    'krn': 'karena',
    'karna': 'karena',
    'dr': 'dari',
    'dri': 'dari',
    'trs': 'terus',
    'trus': 'terus',
    'jd': 'jadi',
    'jdi': 'jadi',
    'bwt': 'buat',
    'utk': 'untuk',
    'blm': 'belum',

    'klu': 'kalau',
    'kalo': 'kalau',
    'klo': 'kalau',
    'kl': 'kalau',
    'kalaw': 'kalau',

    'kyk': 'seperti',
    'kayak': 'seperti',
    'kaya': 'seperti',
    'kek': 'seperti',

    'cm': 'cuma',
    'cmn': 'cuma',
    'cuman': 'cuma',

    'jgn': 'jangan',
    'jngn': 'jangan',

    'gitu': 'begitu',
    'bgtu': 'begitu',
    'gini': 'begini',

    'knp': 'kenapa',
    'ko': 'kok',
    'tak': 'tidak',

    # NEGASI
    'gk': 'tidak',
    'gak': 'tidak',
    'ga': 'tidak',
    'g': 'tidak',
    'nggak': 'tidak',
    'ngga': 'tidak',
    'enggak': 'tidak',
    'tdk': 'tidak',
    'kaga': 'tidak',
    'kagak': 'tidak',
    'ngak': 'tidak',
    'ngk': 'tidak',

    'gada': 'tidak ada',
    'gaada': 'tidak ada',
    'gabisa': 'tidak bisa',
    'gajelas': 'tidak jelas',


    # KATA GANTI ORANG
    'lu': 'kamu',
    'loe': 'kamu',

    'sy': 'saya',
    'sya': 'saya',
    'gw': 'saya',
    'gua': 'saya',
    'gue': 'saya',

    'aq': 'aku',

    'org': 'orang',
    'ank': 'anak',
    'ortu': 'orang tua',
    'mrk': 'mereka',


    # KATA KERJA / BENTUK INFORMAL
    'pake': 'pakai',
    'pke': 'pakai',
    'pkai': 'pakai',
    'pakek': 'pakai',
    'make': 'pakai',
    'dipake': 'dipakai',

    'bgn': 'bangun',

    'dpt': 'dapat',
    'dapet': 'dapat',

    'bkn': 'bukan',

    'bs': 'bisa',
    'bsa': 'bisa',

    'aja': 'saja',
    'ajaa': 'saja',
    'doang': 'saja',

    'liat': 'lihat',
    'kasi': 'kasih',

    'ilang': 'hilang',
    'ngilang': 'hilang',

    'nyesel': 'menyesal',
    'apaan': 'apa',
    'tau': 'tahu',
    'bener': 'benar',
    'benerin': 'perbaiki',
    'nambah': 'tambah',

    'muter': 'berputar',
    'murahin': 'murahkan',
    'adain': 'adakan',


    # KATA SIFAT & KEADAAN
    'baguss': 'bagus',

    'jlek': 'jelek',
    'bapuk': 'jelek',
    'burik': 'buruk',

    'lalot': 'lambat',
    'lelet': 'lambat',
    'lemot': 'lambat',

    'males': 'malas',
    'cepet': 'cepat',

    'tetep': 'tetap',
    'ttp': 'tetap',

    'masi': 'masih',
    'msh': 'masih',

    'sdkt': 'sedikit',
    'dikit': 'sedikit',

    'byk': 'banyak',
    'bnyk': 'banyak',

    'lbh': 'lebih',

    'manisss': 'manis',
    'awallll': 'awal',


    # INTENSIFIER
    'bgd': 'banget',
    'bgt': 'banget',
    'bngt': 'banget',
    'bnget': 'banget',
    'bangett': 'banget',
    'bet': 'banget',
    'bettttt': 'banget',
    'amat': 'banget',


    # SUDAH
    'ud': 'sudah',
    'udh': 'sudah',
    'udah': 'sudah',
    'uda': 'sudah',
    'dah': 'sudah',
    'sdh': 'sudah',


    # LAGI
    'lg': 'lagi',
    'lgy': 'lagi',
    'lgi': 'lagi',


    # WAKTU
    'hr': 'hari',
    'hri': 'hari',
    'thn': 'tahun',
    'bln': 'bulan',

    'skrng': 'sekarang',
    'skrg': 'sekarang',

    'smpe': 'sampai',
    'sampe': 'sampai',

    'ni': 'ini',


    # TEKNIS / PRODUK / JARINGAN
    'abiss': 'habis',
    'abis': 'habis',

    'eror': 'error',

    'aktifa': 'aktif',

    'jaringanx': 'jaringan',
    'jringan': 'jaringan',

    'pitur': 'fitur',

    'lagg': 'lag',
    'ngeleggg': 'lag',
    'ngelag': 'lag',
    'ngeleg': 'lag',

    'kouta': 'kuota',
    'quota': 'kuota',

    'signal': 'sinyal',

    'point': 'poin',

    'rb': 'ribu',

    'abdet': 'update',

    'kepotong': 'terpotong',

    'apk': 'aplikasi',


    # LAIN-LAIN
    'byr': 'bayar',

    'mnrt': 'menurut',
    'pdhl': 'padahal',

    'smoga': 'semoga',
    'moga': 'semoga',

    'gimana': 'bagaimana',

    'yaa': 'ya',
    'yah': 'ya',

    'd': 'di',

    'ad': 'ada',
    'adaaa': 'ada',
    'adaaaaaa': 'ada',

    'tibatiba': 'tiba tiba'
}


# ==========================================
# STOPWORDS
# ==========================================
stopword_id = set(nltk_stopwords.words('indonesian'))

custom_stopwords = {
    'nih', 'btw', 'emang', 'nya', 'ya', 'sih', 'min', 'admin', 'mah', 'pa',
    'deh', 'tuh', 'loh', 'kak', 'iya',
    'lho', 'bos', 'gan', 'hehe', 'haha', 'hehehe'
}

stopword_id.update(custom_stopwords)


# ==========================================
# STEMMER
# ==========================================
factory = StemmerFactory()
stemmer = factory.create_stemmer()


# ==========================================
# PREPROCESSING FUNCTIONS
# ==========================================
def cleaning(text):
    text = str(text)

    # Hapus URL
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Hapus mention
    text = re.sub(r'@\w+', ' ', text)

    # Hapus simbol hashtag, pertahankan katanya
    text = re.sub(r'#(\w+)', r'\1', text)

    # Ganti titik dan underscore menjadi spasi
    text = text.replace('.', ' ')
    text = text.replace('_', ' ')

    # Hapus angka
    text = re.sub(r'\d+', ' ', text)

    # Hapus emoji
    text = re.sub(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+",
        " ",
        text,
        flags=re.UNICODE
    )

    # Hapus tanda baca
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Sisakan huruf dan spasi
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Rapikan spasi
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def case_folding(text):
    return text.lower()


def tokenize(text):
    return text.split()


def normalize_slang(tokens):
    return [
        slang_dict[word] if word in slang_dict else word
        for word in tokens
    ]


def remove_stopwords(tokens):
    return [
        word for word in tokens
        if word not in stopword_id
    ]


def stemming(tokens):
    kalimat = ' '.join(tokens)
    return stemmer.stem(kalimat)


def preprocess(text):
    text = cleaning(text)
    text = case_folding(text)
    tokens = tokenize(text)
    tokens = normalize_slang(tokens)
    tokens = remove_stopwords(tokens)
    text = stemming(tokens)

    return text


# ==========================================
# STREAMLIT UI
# ==========================================
st.title("📱 Sentiment Analysis MyXL")

st.write(
    "Masukkan ulasan mengenai layanan MyXL untuk memprediksi "
    "sentimen Positive, Neutral, atau Negative."
)

user_input = st.text_area(
    "Masukkan ulasan:",
    placeholder="Contoh: Jaringannya lemot banget dan aplikasinya sering error"
)


# ==========================================
# PREDICTION
# ==========================================
if st.button("Prediksi Sentimen"):

    if user_input.strip() == "":
        st.warning("Masukkan ulasan terlebih dahulu.")

    else:
        # Preprocessing
        processed_text = preprocess(user_input)

        # TF-IDF Transform
        text_tfidf = tfidf.transform([processed_text])

        # Prediksi sentimen
        prediction = model.predict(text_tfidf)[0]

        # Probabilitas setiap kelas
        probabilities = model.predict_proba(text_tfidf)[0]

        # Ambil probabilitas kelas yang diprediksi
        predicted_index = list(model.classes_).index(prediction)
        confidence = probabilities[predicted_index] * 100

        # Tampilkan hasil
        st.subheader("Hasil Prediksi")

        if prediction == "positive":
            st.success("Sentimen: Positive 😊💖")

        elif prediction == "negative":
            st.error("Sentimen: Negative 😞💔")

        else:
            st.info("Sentimen: Neutral 😶🎭")

        # Tampilkan confidence score
        st.write(f"Confidence Score: **{confidence:.2f}%**")
        st.progress(float(confidence / 100))

        # Tampilkan hasil preprocessing
        with st.expander("Lihat hasil preprocessing"):
            st.write(processed_text)
