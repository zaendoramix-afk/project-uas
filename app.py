import os
import uuid
import json
import shutil
import time
from threading import Thread

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, send_file, jsonify
)
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("PIXELFORGE_SECRET_KEY", os.urandom(24).hex())
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB

UPLOAD_FOLDER = os.path.join("static", "uploads")
RESULT_FOLDER = os.path.join("static", "results")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# ---------------------------------------------------------------------------
# Fungsi Tambahan - Pembersih File Sampah Otomatis (Anti-Storage Bloat)
# ---------------------------------------------------------------------------
def cleanup_old_sessions(interval_seconds=3600, max_age_seconds=7200):
    """Menghapus folder sesi DAN file sampah satuan yang sudah tidak aktif."""
    first_run = True
    
    while True:
        if not first_run:
            time.sleep(interval_seconds)
        first_run = False
        
        now = time.time()
        for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
            if not os.path.exists(folder):
                continue
            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)
                
                # 1. Jika itu FOLDER sesi lama (lebih dari 2 jam)
                if os.path.isdir(item_path):
                    if now - os.path.getmtime(item_path) > max_age_seconds:
                        try:
                            shutil.rmtree(item_path)
                            print(f"🧹 [Auto-Cleanup] Folder sampah berhasil dihapus: {item_path}")
                        except Exception as e:
                            print(f"❌ Gagal menghapus folder {item_path}: {e}")
                            
                # 2. Jika itu FILE gambar satuan yang telanjur tercecer di luar folder
                elif os.path.isfile(item_path):
                    if now - os.path.getmtime(item_path) > max_age_seconds:
                        try:
                            os.remove(item_path)
                            print(f"🧹 [Auto-Cleanup] File gambar tercecer berhasil dihapus: {item_path}")
                        except Exception as e:
                            print(f"❌ Gagal menghapus file {item_path}: {e}")

# Jalankan pembersih otomatis di background thread
cleanup_thread = Thread(target=cleanup_old_sessions, daemon=True)
cleanup_thread.start()


# ---------------------------------------------------------------------------
# Helper - Sesi & File
# ---------------------------------------------------------------------------

def get_session_id():
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return session["sid"]


def session_upload_dir():
    path = os.path.join(UPLOAD_FOLDER, get_session_id())
    os.makedirs(path, exist_ok=True)
    return path


def session_result_dir():
    path = os.path.join(RESULT_FOLDER, get_session_id())
    os.makedirs(path, exist_ok=True)
    return path


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_real_image(filepath):
    try:
        with open(filepath, "rb") as f:
            header = f.read(12)
    except OSError:
        return False

    is_png = header.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpeg = header.startswith(b"\xff\xd8\xff")
    if not (is_png or is_jpeg):
        return False

    return cv2.imread(filepath) is not None


def to_web_path(path):
    return "/" + path.replace("\\", "/")


def load_current_image():
    current = session.get("current_image")
    if not current or not os.path.exists(current):
        return None
    img = cv2.imread(current, cv2.IMREAD_UNCHANGED)
    return img


def save_result(img, prefix):
    filename = f"{prefix}_{uuid.uuid4().hex[:10]}.jpg"
    result_path = os.path.join(session_result_dir(), filename)
    cv2.imwrite(result_path, img)
    session["last_result"] = result_path
    return to_web_path(result_path)


def save_plot_result(fig, prefix):
    filename = f"{prefix}_{uuid.uuid4().hex[:10]}.png"
    result_path = os.path.join(session_result_dir(), filename)
    fig.savefig(result_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    session["last_result"] = result_path
    return to_web_path(result_path)


# ---------------------------------------------------------------------------
# Operasi Citra (Diproteksi dari Error Dimensi / Channel)
# ---------------------------------------------------------------------------

def ensure_bgr(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def op_grayscale(img, **kwargs):
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def op_negative(img, **kwargs):
    return 255 - img


def op_binary(img, threshold=127, **kwargs):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def op_brightness(img, brightness_value=0, **kwargs):
    return cv2.convertScaleAbs(img, alpha=1, beta=brightness_value)


def op_contrast(img, contrast_value=1.0, **kwargs):
    return cv2.convertScaleAbs(img, alpha=contrast_value, beta=0)


def op_saturation(img, saturation_value=1.0, **kwargs):
    img_bgr = ensure_bgr(img)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_value, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def op_sepia(img, intensity=1.0, **kwargs):
    img_bgr = ensure_bgr(img)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
    ])
    sepia = img_rgb @ sepia_matrix.T
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    
    result = cv2.cvtColor(sepia, cv2.COLOR_RGB2BGR)
    if intensity < 1.0:
        result = cv2.addWeighted(img_bgr, 1 - intensity, result, intensity, 0)
    return result


def op_sketch(img, **kwargs):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blur = 255 - blurred
    return cv2.divide(gray, inverted_blur, scale=256.0)


def op_emboss(img, **kwargs):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    embossed = cv2.filter2D(gray, -1, kernel)
    return cv2.add(embossed, 128)


def op_blur(img, blur_strength=15, **kwargs):
    kernel_size = max(3, blur_strength)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def op_sharpen(img, sharpen_strength=1.0, **kwargs):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    if sharpen_strength < 1.0:
        sharpened = cv2.addWeighted(img, 1 - sharpen_strength, sharpened, sharpen_strength, 0)
    return sharpened


def op_vignette(img, vignette_strength=1.0, **kwargs):
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols / 2.8)
    kernel_y = cv2.getGaussianKernel(rows, rows / 2.8)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    mask = (mask ** (1.0 / vignette_strength)) if vignette_strength > 0 else mask
    
    if len(img.shape) == 3:
        out = img.astype(np.float64)
        for c in range(3):
            out[:, :, c] *= mask
        return out.astype(np.uint8)
    else:
        return (img.astype(np.float64) * mask).astype(np.uint8)


def op_rotate(img, angle=90, **kwargs):
    rows, cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, rotation_matrix, (cols, rows))


def op_flip_h(img, **kwargs):
    return cv2.flip(img, 1)


def op_flip_v(img, **kwargs):
    return cv2.flip(img, 0)


def op_zoom_in(img, zoom_level=1.5, **kwargs):
    h_orig, w_orig = img.shape[:2]
    img_zoomed = cv2.resize(img, None, fx=zoom_level, fy=zoom_level, interpolation=cv2.INTER_LINEAR)
    h_zoomed, w_zoomed = img_zoomed.shape[:2]
    start_y = (h_zoomed - h_orig) // 2
    start_x = (w_zoomed - w_orig) // 2
    cropped = img_zoomed[start_y:start_y + h_orig, start_x:start_x + w_orig]
    return cropped


def op_zoom_out(img, zoom_level=2.0, **kwargs):
    height, width = img.shape[:2]
    new_width = max(1, int(width / zoom_level))
    new_height = max(1, int(height / zoom_level))
    return cv2.resize(img, (new_width, new_height))


def op_edge(img, threshold1=100, threshold2=200, **kwargs):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return cv2.Canny(gray, threshold1, threshold2)


IMAGE_OPERATIONS = {
    "grayscale": ("grayscale", op_grayscale, {}),
    "negative": ("negative", op_negative, {}),
    "binary": ("binary", op_binary, {"threshold": 127}),
    "brightness": ("brightness", op_brightness, {"brightness_value": 0}),
    "contrast": ("contrast", op_contrast, {"contrast_value": 1.0}),
    "saturation": ("saturation", op_saturation, {"saturation_value": 1.0}),
    "sepia": ("sepia", op_sepia, {"intensity": 1.0}),
    "sketch": ("sketch", op_sketch, {}),
    "emboss": ("emboss", op_emboss, {}),
    "blur": ("blur", op_blur, {"blur_strength": 15}),
    "sharpen": ("sharpen", op_sharpen, {"sharpen_strength": 1.0}),
    "vignette": ("vignette", op_vignette, {"vignette_strength": 1.0}),
    "rotate": ("rotate", op_rotate, {"angle": 90}),
    "flip_h": ("flip_h", op_flip_h, {}),
    "flip_v": ("flip_v", op_flip_v, {}),
    "zoom_in": ("zoom_in", op_zoom_in, {"zoom_level": 1.5}),
    "zoom_out": ("zoom_out", op_zoom_out, {"zoom_level": 2.0}),
    "edge": ("edge", op_edge, {"threshold1": 100, "threshold2": 200}),
}


def render_histogram(img):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#10131c")
    ax.set_facecolor("#10131c")
    
    if len(img.shape) == 3:
        colors = (("b", "#3b82f6"), ("g", "#22c55e"), ("r", "#ef4444"))
        for i, (_, hex_color) in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            ax.plot(hist, color=hex_color, linewidth=1.4)
    else:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        ax.plot(hist, color="#94a3b8", linewidth=1.4)
        
    ax.set_xlim([0, 256])
    ax.set_title("Histogram Warna", color="#e2e8f0")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    return save_plot_result(fig, "histogram")


def render_fourier(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#10131c")
    # Mengubah cmap dari "magma" menjadi "gray" agar kembali abu-abu
    ax.imshow(magnitude, cmap="gray")
    ax.set_title("Spektrum Fourier", color="#e2e8f0")
    ax.axis("off")
    return save_plot_result(fig, "fourier")


# ---------------------------------------------------------------------------
# Route Utama
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    get_session_id()

    original_image = None
    result_image = None
    image_info = None
    error_message = None
    success_message = None

    if request.method == "POST":
        # Fleksibel membaca action baik dari AJAX JSON maupun Form Submit biasa
        if request.is_json:
            action = request.json.get("action")
            form_data = request.json
        else:
            action = request.form.get("action")
            form_data = request.form

        if action == "upload":
            if "image" not in request.files or request.files["image"].filename == "":
                error_message = "Pilih file gambar terlebih dahulu."
            else:
                file = request.files["image"]
                if not allowed_file(file.filename):
                    error_message = "Format file tidak didukung. Gunakan PNG, JPG, atau JPEG."
                else:
                    filename = f"{uuid.uuid4().hex[:10]}_{secure_filename(file.filename)}"
                    filepath = os.path.join(session_upload_dir(), filename)
                    file.save(filepath)

                    if not is_real_image(filepath):
                        os.remove(filepath)
                        error_message = "File ini bukan gambar yang valid (isi file tidak sesuai)."
                    else:
                        session["current_image"] = filepath
                        session.pop("last_result", None)
                        success_message = "Gambar berhasil diunggah."

        elif action == "reset":
            sid = session.get("sid")
            if sid:
                user_upload_dir = os.path.join(UPLOAD_FOLDER, sid)
                user_result_dir = os.path.join(RESULT_FOLDER, sid)
                
                if os.path.exists(user_upload_dir):
                    try: shutil.rmtree(user_upload_dir)
                    except Exception as e: print(f"Gagal menghapus folder upload: {e}")
                        
                if os.path.exists(user_result_dir):
                    try: shutil.rmtree(user_result_dir)
                    except Exception as e: print(f"Gagal menghapus folder hasil: {e}")
            
            session.pop("current_image", None)
            session.pop("last_result", None)
            success_message = "Gambar berhasil dihapus dari server."
            
            if request.is_json:
                return jsonify({"success": True, "message": "File di server berhasil dihapus permanen"})

        elif action == "use_result":
            last_result = session.get("last_result")
            if last_result and os.path.exists(last_result):
                if "histogram" in last_result or "fourier" in last_result:
                    error_message = "Hasil terakhir berupa grafik analisis, tidak bisa ditumpuk efek."
                else:
                    new_path = os.path.join(session_upload_dir(), f"{uuid.uuid4().hex[:10]}_chained.jpg")
                    img = cv2.imread(last_result, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        cv2.imwrite(new_path, img)
                        session["current_image"] = new_path
                        success_message = "Hasil sekarang jadi gambar utama. Tumpuk efek lain di atasnya."
                    else:
                        error_message = "Gagal memproses file hasil terakhir."
            else:
                error_message = "Belum ada hasil untuk dijadikan gambar utama."

        elif action in IMAGE_OPERATIONS or action in ("histogram", "fourier"):
            img = load_current_image()

            if img is None and session.get("current_image"):
                error_message = "Gambar tidak dapat dibaca atau filenya rusak."
                session.pop("current_image", None)
            elif img is None:
                error_message = "Belum ada gambar yang diunggah."
            elif action == "histogram":
                result_image = render_histogram(img)
            elif action == "fourier":
                result_image = render_fourier(img)
            else:
                prefix, func, defaults = IMAGE_OPERATIONS[action]
                params = defaults.copy()
                
                # Mengambil data form_data secara dinamis & aman dari kegagalan tipe data
                for key in params.keys():
                    raw_val = form_data.get(key)
                    if raw_val is not None and raw_val != "":
                        try:
                            if isinstance(params[key], float):
                                params[key] = float(raw_val)
                            elif isinstance(params[key], int):
                                params[key] = int(raw_val)
                        except ValueError:
                            pass
                
                try:
                    processed = func(img, **params)
                    result_image = save_result(processed, prefix)
                except Exception as e:
                    error_message = f"Gagal memproses gambar: {str(e)}"

    current_image = session.get("current_image")
    if current_image and os.path.exists(current_image):
        original_image = to_web_path(current_image)
        img = cv2.imread(current_image, cv2.IMREAD_UNCHANGED)
        if img is not None:
            height, width = img.shape[:2]
            channels = img.shape[2] if len(img.shape) == 3 else 1
            sample_pixel = img[0, 0].tolist()
            file_size_kb = round(os.path.getsize(current_image) / 1024, 1)
            image_info = {
                "filename": os.path.basename(current_image).split("_", 1)[-1],
                "width": width,
                "height": height,
                "channels": channels,
                "pixel": sample_pixel,
                "size_kb": file_size_kb,
            }

    has_result = bool(session.get("last_result")) and os.path.exists(session.get("last_result", ""))
    last_res_path = session.get("last_result", "")
    is_chart = "histogram" in last_res_path or "fourier" in last_res_path
    can_chain = has_result and not is_chart

    return render_template(
        "index.html",
        original_image=original_image,
        result_image=result_image,
        image_info=image_info,
        error_message=error_message,
        success_message=success_message,
        has_result=has_result,
        can_chain=can_chain,
    )


@app.route("/api/process", methods=["POST"])
def api_process():
    try:
        action = request.json.get("action")
        params = request.json.get("params", {})
        
        if action not in IMAGE_OPERATIONS and action not in ("histogram", "fourier"):
            return jsonify({"success": False, "error": "Aksi tidak dikenali"}), 400
        
        img = load_current_image()
        if img is None:
            return jsonify({"success": False, "error": "Belum ada gambar yang diunggah"}), 400
        
        if action == "histogram":
            result_path = render_histogram(img)
        elif action == "fourier":
            result_path = render_fourier(img)
        else:
            prefix, func, defaults = IMAGE_OPERATIONS[action]
            op_params = defaults.copy()
            op_params.update(params)
            
            processed = func(img, **op_params)
            result_path = save_result(processed, prefix)
        
        return jsonify({
            "success": True,
            "result_path": result_path
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/download")
def download():
    last_result = session.get("last_result")
    if not last_result or not os.path.exists(last_result):
        return redirect(url_for("index"))
    
    abs_result_folder = os.path.abspath(RESULT_FOLDER)
    abs_target_file = os.path.abspath(last_result)
    if not abs_target_file.startswith(abs_result_folder):
        return jsonify({"error": "Akses ditolak"}), 403

    download_name = "pixelforge_" + os.path.basename(last_result)
    return send_file(last_result, as_attachment=True, download_name=download_name)


if __name__ == "__main__":
    app.run(debug=True)