import json
import sys
import requests
from PyQt5.QtCore import Qt, QRectF, QUrl
from PyQt5.QtGui import QBrush, QColor, QPen, QPixmap, QDesktopServices
from PyQt5.QtWebSockets import QWebSocket
from PyQt5.QtWidgets import (QApplication, QFileDialog, QGraphicsRectItem,
                             QGraphicsScene, QGraphicsView, QHBoxLayout,
                             QInputDialog, QLabel, QLineEdit, QMainWindow,
                             QMessageBox, QPushButton, QVBoxLayout, QWidget,
                             QProgressBar, QComboBox)
from passlib.context import CryptContext
from fastapi import HTTPException, Form

COLOR_VETO = "#673ab7"  # Fioletowy
COLOR_HIGH_CONF = "#388e3c"  # Zielony
COLOR_LOW_CONF = "#f57c00"  # Pomarańczowy
COLOR_UNCERTAIN = QColor(255, 235, 59, 100)  # Żółty przezroczysty

CONFIG_FILE = Path.home() / ".htr_client_config"

def get_api_url():
    """ Pobiera URL do configu. """
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return f.read().strip()
    return "http://localhost:8000"

def save_api_url(url):
    """ Zapisuje config po zmianach. """
    with open(CONFIG_FILE, "w") as f:
        f.write(url)

# Konfiguracja szyfrowania haseł
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")



@app.post("/token")
async def login(username: str = Form(...), password: str = Form(...)):
    """ Logowanie w oparciu o MongoDB, zapewnia indywidualne konto per użytkownik. """

    # Szukamy użytkownika w bazie
    try:
        user = await state["db"].users.find_one({"username": username})
    except Exception as e:
        logger.error(f"Błąd bazy danych podczas logowania: {e}")
        raise HTTPException(status_code=500, detail="Błąd wewnętrzny serwera")

    # Sprawdzamy, czy użytkownik istnieje
    if not user:
        raise HTTPException(status_code=401, detail="Nieprawidłowy login lub hasło")

    # Weryfikujemy hasło
    if not pwd_context.verify(password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Nieprawidłowy login lub hasło")

    # Generujemy i zwracamy token
    return {
        "access_token": str(user["username"]),
        "token_type": "bearer",
        "role": user.get("role", "user")
    }


class WordBoxItem(QGraphicsRectItem):
    """ Ramka słowa z dynamicznym Tooltipem i podświetlaniem fragmentów niepewnych. """
    def __init__(self, x, y, w, h, meta, main_window=None, document_id=None):
        super().__init__(QRectF(x, y, w, h))
        self.main_window = main_window
        self.document_id = document_id
        self.current_text = meta.get('final_result', '')

        # Logika wizualna ramki
        conf = meta.get('confidence', 0.0)
        is_veto = meta.get('veto_active', False)
        is_onnx = meta.get('mode') == 'onnx'

        # Wybór koloru na podstawie stanu metadanych
        color = COLOR_VETO if is_veto else (COLOR_HIGH_CONF if conf > 0.85 else COLOR_LOW_CONF)

        self.setPen(QPen(QColor(color), 2 if not is_veto else 3))
        self.setBrush(QBrush(QColor(color).lighter(170), 40))
        self.setAcceptHoverEvents(True)

        # Budowa Tooltipa zgodnie z wymaganiami (crnn result, final result itd.)
        engine_str = "<span style='color:blue;'>ONNX</span>" if is_onnx else "<span style='color:green;'>Cascade</span>"
        veto_tag = "<br><b style='color:red;'>[VETO]</b>" if is_veto else ""

        self.tooltip_template = (
            f"<div style='font-family: Arial;'>"
            f"<b>Silnik:</b> {engine_str}{veto_tag}<br><hr>"
            f"<b>Propozycja CRNN:</b> {meta.get('crnn_result', '-')}<br>"
            f"<b>Po poprawkach CapsNet:</b> {meta.get('capsnet_result', '-')}<br>"
            f"<b>Pewność:</b> {conf * 100:.1f}%<br><hr>"
            f"<b>Wynik końcowy:</b> <span style='font-size:14px;'>{{final_res}}</span>"
            f"</div>"
        )
        self.update_tooltip()

        # Obsługa stref niepewności (uncertain zones)
        self.uncertain_rects = []
        for zone in meta.get('uncertain_zones', []):
            zx = x + (zone['x'] * w)
            zw = zone['w'] * w
            rect = QGraphicsRectItem(QRectF(zx, y, zw, h), parent=self)
            rect.setBrush(QBrush(COLOR_UNCERTAIN))
            rect.setPen(Qt.PenStyle.NoPen)
            rect.hide()
            self.uncertain_rects.append(rect)

    def update_tooltip(self):
        self.setToolTip(self.tooltip_template.format(final_res=self.current_text))

    def hoverEnterEvent(self, event):
        for r in self.uncertain_rects: r.show()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        for r in self.uncertain_rects: r.hide()
        super().hoverLeaveEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.main_window: self.main_window.prompt_feedback(self)

    def update_result(self, new_text):
        self.current_text = new_text
        self.update_tooltip()
        self.setPen(QPen(QColor(COLOR_HIGH_CONF), 2))
        self.setBrush(QBrush(QColor(COLOR_HIGH_CONF).lighter(170), 60))


class HTRMainWindow(QMainWindow):
    """ Główny panel sterowania logiką. """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kaskadowy System HTR - Panel Eksperymentalny")
        self.setGeometry(100, 100, 1100, 850)

        # Konfiguracja API
        self.api_url = "http://localhost:8000"
        self.access_token = None  # Tu trzymamy nasz klucz do królestwa
        self.file_list = []
        self.current_file_idx = 0
        self.current_task_id = None
        self.session_doc_ids = []

        # Komponenty interfejsu
        self.user_id_input = None
        self.password_input = None
        self.lang_combo = None
        self.progress_bar = None
        self.btn_adapt = None
        self.scene = None
        self.view = None
        self.btn_load = None
        self.btn_next = None
        self.btn_pdf = None
        self.status = None
        self.btn_login = None

        self.init_ui()

        # Websocket do zarządzania aktualizacjami stanu zadania
        self.websocket = QWebSocket()
        self.websocket.textMessageReceived.connect(self.on_websocket_message)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Panel górny: Logowanie, Adaptacja i Język
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Login:"))

        self.user_id_input = QLineEdit("user_01")
        self.user_id_input.setFixedWidth(80)
        top_bar.addWidget(self.user_id_input)

        # Pole na hasło
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setPlaceholderText("Hasło")
        self.password_input.setFixedWidth(100)
        top_bar.addWidget(self.password_input)

        # Przycisk logowania
        self.btn_login = QPushButton("Zaloguj")
        self.btn_login.clicked.connect(self.login_to_api)
        top_bar.addWidget(self.btn_login)

        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["Polski", "English"])
        top_bar.addWidget(self.lang_combo)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 15)
        top_bar.addWidget(self.progress_bar)

        self.btn_adapt = QPushButton("Adaptuj Model")
        self.btn_adapt.clicked.connect(self.trigger_adaptation)
        top_bar.addWidget(self.btn_adapt)
        layout.addLayout(top_bar)

        # Widok graficzny
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)

        self.view.setMouseTracking(True)
        viewport = self.view.viewport()
        if viewport is not None:
            viewport.setMouseTracking(True)

        layout.addWidget(self.view)

        # Przyciski sterowania
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("Wczytaj Obrazy")
        self.btn_load.clicked.connect(self.load_images)
        self.btn_load.setEnabled(False)  # Zablokowane do czasu zalogowania
        btn_layout.addWidget(self.btn_load)

        self.btn_next = QPushButton("Analizuj Stronę")
        self.btn_next.clicked.connect(self.process_page)
        self.btn_next.setEnabled(False)
        btn_layout.addWidget(self.btn_next)

        self.btn_pdf = QPushButton("Generuj Raport PDF")
        self.btn_pdf.clicked.connect(self.download_final_pdf)
        btn_layout.addWidget(self.btn_pdf)
        layout.addLayout(btn_layout)

        self.status = QLabel("Oczekuję na logowanie.")
        layout.addWidget(self.status)

    def login_to_api(self):
        """ Autoryzacja i pobranie tokena z serwera. """
        try:
            data = {
                "username": self.user_id_input.text(),
                "password": self.password_input.text()
            }
            # OAuth2 w FastAPI wymaga zwykłego wysłania 'data' (x-www-form-urlencoded)
            r = requests.post(f"{self.api_url}/token", data=data)

            if r.status_code == 200:
                self.access_token = r.json().get("access_token")
                self.status.setText(f"Zalogowano jako {data['username']}. System gotowy.")
                self.btn_load.setEnabled(True)  # Odblokowujemy aplikację
            else:
                self.status.setText("Błąd logowania. Nieprawidłowe hasło?")
                QMessageBox.warning(self, "Błąd", "Logowanie nie powiodło się.")
        except Exception as e:
            self.status.setText(f"Błąd połączenia z serwerem: {e}")

    def load_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Wybierz skany", "", "Images (*.png *.jpg)")
        if files:
            self.file_list = sorted(files)
            self.current_file_idx = 0
            self.status.setText(f"Załadowano {len(self.file_list)} plików.")
            self.btn_next.setEnabled(True)

    def process_page(self):
        if not self.access_token:
            QMessageBox.warning(self, "Błąd", "Musisz się najpierw zalogować!")
            return

        if self.current_file_idx >= len(self.file_list):
            return

        path = self.file_list[self.current_file_idx]
        self.scene.clear()
        self.scene.addPixmap(QPixmap(path))

        try:
            self.status.setText("Wysyłanie do chmury.")
            self.btn_next.setEnabled(False)

            with open(path, 'rb') as f:
                files = {'file': f}
                # Usunęliśmy user_id, bo teraz idzie w tokenie
                data = {'language': "pl" if self.lang_combo.currentIndex() == 0 else "en"}

                headers = {"Authorization": f"Bearer {self.access_token}"}
                r = requests.post(f"{self.api_url}/process-document", files=files, data=data, headers=headers)

            if r.status_code == 200:
                self.current_task_id = r.json().get('task_id')

                # Dynamiczny adres WebSocket
                ws_base = self.api_url.replace("https://", "ws://").replace("https://", "wss://")
                ws_url = f"{ws_base}/ws/status/{self.current_task_id}"

                self.websocket.open(QUrl(ws_url))
                self.status.setText("Czekam na wyniki z GPU (strumieniowanie na żywo).")
            else:
                self.status.setText(f"Błąd API: {r.status_code}")
                self.btn_next.setEnabled(True)
        except Exception as e:
            self.status.setText(f"Błąd połączenia: {str(e)}")
            self.btn_next.setEnabled(True)

    def on_websocket_message(self, message):
        try:
            data = json.loads(message)

            if data.get('status') == 'SUCCESS':
                self.websocket.close()
                self.render_results(data.get('result'))
                self.current_file_idx += 1
                self.btn_next.setEnabled(True)
                self.status.setText("Analiza gotowa. (Zero opóźnień!)")

            elif data.get('status') == 'FAILED':
                self.websocket.close()
                self.btn_next.setEnabled(True)
                self.status.setText("Wystąpił błąd podczas analizy na GPU.")

        except Exception as e:
            print(f"Błąd przetwarzania wiadomości WebSocket: {e}")

    def render_results(self, data):
        doc_id = data.get('document_id')
        self.session_doc_ids.append(doc_id)
        for w in data.get('words', []):
            x, y, width, height = w['box']
            item = WordBoxItem(x, y, width, height, w, self, doc_id)
            self.scene.addItem(item)

    def prompt_feedback(self, box):
        text, ok = QInputDialog.getText(self, "Korekta", "Popraw tekst:", QLineEdit.EchoMode.Normal, box.current_text)
        if ok and text:
            try:
                box_rect = box.rect()
                payload = {
                    'document_id': box.document_id,
                    'correct_text': text,
                    'box': {
                        'x': box_rect.x(),
                        'y': box_rect.y(),
                        'w': box_rect.width(),
                        'h': box_rect.height()
                    }
                }

                headers = {"Authorization": f"Bearer {self.access_token}"}
                response = requests.post(f"{self.api_url}/submit-feedback", json=payload, headers=headers, timeout=10)
                response.raise_for_status()

                box.update_result(text)
                self.progress_bar.setValue(min(self.progress_bar.value() + 1, 15))
                self.status.setText("Korekta została pomyślnie wysłana i zapisana.")

            except Exception as e:
                print(f"Błąd feedbacku: {e}")
                self.status.setText("Wystąpił błąd podczas wysyłania korekty.")

    def trigger_adaptation(self):
        if not self.access_token:
            QMessageBox.warning(self, "Błąd", "Zaloguj się najpierw.")
            return

        headers = {"Authorization": f"Bearer {self.access_token}"}

        # Teraz endpoint bierze usera z tokenu, nie ma potrzeby wysyłać go w body/URL
        requests.post(f"{self.api_url}/trigger-adaptation", headers=headers)
        QMessageBox.information(self, "Adaptacja", "Proces adaptacji wag rozpoczęty w tle.")

    def download_final_pdf(self):
        if not self.session_doc_ids: return
        if not self.access_token: return

        headers = {"Authorization": f"Bearer {self.access_token}"}
        r = requests.post(f"{self.api_url}/generate-pdf", json={'document_ids': self.session_doc_ids}, headers=headers)

        if r.status_code == 200:
            metrics = r.json().get('metrics', {})
            msg = f"CER: {metrics.get('cer'):.2f}%\nWER: {metrics.get('avg_confidence'):.2f}%"
            QMessageBox.information(self, "Statystyki Sesji", msg)
            QDesktopServices.openUrl(QUrl(r.json().get('download_url')))

    def change_server_address(self):
        new_url, ok = QInputDialog.getText(
            self, "Ustawienia Sieciowe",
            "Adres serwera API (wymagane https://, np. https://api.twojafirma.pl):",
            QLineEdit.EchoMode.Normal, self.api_url
        )
        if ok and new_url:
            new_url = new_url.strip()

            # Walidacja bezpieczeństwa
            if new_url.startswith("https://") and "localhost" not in new_url and "127.0.0.1" not in new_url:
                reply = QMessageBox.warning(
                    self, "Zagrożenie Bezpieczeństwa",
                    "Próbujesz połączyć się przez niezabezpieczony protokół HTTP.\nTwoje hasło może zostać przechwycone!\n\nCzy na pewno chcesz kontynuować?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return

            self.api_url = new_url
            save_api_url(self.api_url)
            self.status.setText(f"Adres serwera zmieniony na: {self.api_url}")
            QMessageBox.information(self, "Sukces", "Adres API został zaktualizowany.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ex = HTRMainWindow()
    ex.show()
    sys.exit(app.exec_())