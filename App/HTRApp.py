import os
import csv
from datetime import datetime
import json
import sys
import requests
import random
import Levenshtein
from pathlib import Path
from PyQt5.QtCore import Qt, QRectF, QUrl, QTimer
from PyQt5.QtGui import QBrush, QColor, QPen, QPixmap, QDesktopServices
from PyQt5.QtWebSockets import QWebSocket
from PyQt5.QtWidgets import (QApplication, QFileDialog, QGraphicsRectItem,
                             QGraphicsScene, QGraphicsView, QHBoxLayout,
                             QInputDialog, QLabel, QLineEdit, QMainWindow,
                             QMessageBox, QPushButton, QVBoxLayout, QWidget,
                             QProgressBar, QComboBox, QCheckBox,
                             QDialog, QGridLayout, QToolButton)

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


class WordBoxItem(QGraphicsRectItem):
    """ Ramka słowa z dynamicznym Tooltipem i podświetlaniem fragmentów niepewnych. """
    def __init__(self, x, y, w, h, meta, main_window=None, document_id=None):
        super().__init__(QRectF(x, y, w, h))
        self.main_window = main_window
        self.document_id = document_id
        self.word_id = meta.get('word_id', '')
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

    def contextMenuEvent(self, event):
        from PyQt5.QtWidgets import QMenu
        menu = QMenu()
        
        merge_action = menu.addAction("Scal zaznaczone")
        split_action = menu.addAction("Podziel")
        
        action = menu.exec_(event.screenPos())
        
        if action == merge_action:
            if self.main_window: self.main_window.merge_selected_boxes()
        elif action == split_action:
            if self.main_window: self.main_window.split_box(self)


class MemoryGameDialog(QDialog):
    """ Mini-gra zabijająca czas podczas długiej inferencji kaskady HTR. """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analiza przesłanego tekstu trwa. Zagraj w Memory!")
        self.setFixedSize(320, 320)
        
        # Blokuje klikanie w główne okno, ale pozwala działać WebSocketom w tle
        self.setWindowModality(Qt.WindowModal) 
        
        self.layout = QGridLayout(self)
        self.emojis = ['🤖', '🤖', '📝', '📝', '🔍', '🔍', '🧠', '🧠', '⚡', '⚡', '📚', '📚']
        random.shuffle(self.emojis)
        
        self.buttons = []
        self.first_btn = None
        self.matches = 0
        self.lock = False

        for i in range(12):
            btn = QToolButton()
            btn.setFixedSize(70, 70)
            btn.setStyleSheet("font-size: 30px; background-color: #333; border-radius: 8px;")
            btn.clicked.connect(lambda checked, idx=i: self.on_card_click(idx))
            self.layout.addWidget(btn, i // 4, i % 4)
            self.buttons.append(btn)

    def on_card_click(self, idx):
        if self.lock or self.buttons[idx].text() != "":
            return
            
        btn = self.buttons[idx]
        btn.setText(self.emojis[idx])
        btn.setStyleSheet("font-size: 30px; background-color: #fff; color: #000; border-radius: 8px;")

        if self.first_btn is None:
            self.first_btn = (idx, btn)
        else:
            self.lock = True
            first_idx, first_b = self.first_btn
            
            if self.emojis[first_idx] == self.emojis[idx]:
                self.matches += 1
                self.first_btn = None
                self.lock = False
                if self.matches == 6:
                    self.setWindowTitle("Wygrana! Czekam na resztę danych.")
            else:
                QTimer.singleShot(800, lambda: self.hide_cards(first_b, btn))

    def hide_cards(self, b1, b2):
        for b in (b1, b2):
            b.setText("")
            b.setStyleSheet("font-size: 30px; background-color: #333; border-radius: 8px;")
        self.first_btn = None
        self.lock = False


class HTRMainWindow(QMainWindow):
    """ Główny panel sterowania logiką. """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kaskadowy System HTR - Panel Eksperymentalny")
        self.setGeometry(100, 100, 1100, 850)

        # Konfiguracja API
        self.api_url = get_api_url() 
        self.access_token = None
        self.file_list = []
        self.current_file_idx = 0
        self.current_task_id = None
        self.session_doc_ids = []
        self.memory_game = None # Referencja do okna gry

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

        # Zmienne do Live Evaluation podczas sesji
        self.total_chars = 0
        self.total_char_errors = 0
        self.total_words = 0
        self.total_word_errors = 0

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

        # Zabezpieczenie przed brakiem inicjalizacji zmiennej
        if not hasattr(self, 'password_input') or self.password_input is None:
            self.password_input = QLineEdit()
            
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
        
        # Przełącznik trybu testowego
        self.test_mode_cb = QCheckBox("Tryb testowy")
        top_bar.addWidget(self.test_mode_cb)

        # Opcjonalna Analiza Układu (Tabele/Kolumny)
        self.layout_cb = QCheckBox("Złożony układ (Tabele/Kolumny)")
        top_bar.addWidget(self.layout_cb)

        # Miejsce na metryki sesji (domyślnie ukryte)
        self.metrics_label = QLabel("Metryki sesji: Czekam na pierwszą korektę.")
        self.metrics_label.setStyleSheet("color: gray; font-size: 11px; margin-left: 10px;")
        self.metrics_label.hide() 
        top_bar.addWidget(self.metrics_label)
        
        # Przycisk resetowania metryk sesji testowej
        self.btn_reset_metrics = QPushButton("Resetuj")
        self.btn_reset_metrics.setStyleSheet("font-size: 10px; padding: 2px;")
        self.btn_reset_metrics.hide()
        self.btn_reset_metrics.clicked.connect(self.reset_metrics)
        top_bar.addWidget(self.btn_reset_metrics)

        # Odkryj/ukryj elementy, gdy klikamy checkbox
        self.test_mode_cb.stateChanged.connect(
            lambda state: (self.metrics_label.setVisible(state == Qt.Checked),
                           self.btn_reset_metrics.setVisible(state == Qt.Checked))
        )

        # Pasek do głównego ekranu
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
        
        self.btn_stats = QPushButton("Pokaż Statystyki Sesji")
        self.btn_stats.setToolTip("Pokazuje aktualne CER/WER dla tej sesji.")
        self.btn_stats.clicked.connect(self.display_session_summary)
        btn_layout.addWidget(self.btn_stats)

        self.btn_server = QPushButton("Zmień Serwer API")
        self.btn_server.clicked.connect(self.change_server_address)
        btn_layout.addWidget(self.btn_server)

        layout.addLayout(btn_layout)

        self.status = QLabel("Oczekuję na logowanie.")
        layout.addWidget(self.status)

    def reset_metrics(self):
        """ Zeruje statystyki sesji testowej. """
        self.total_chars = 0
        self.total_char_errors = 0
        self.total_words = 0
        self.total_word_errors = 0
        self.metrics_label.setText("CER: 0.00% | WER: 0.00%")
        self.status.setText("Statystyki sesji testowej zostały zresetowane.")

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

                # Wysyłanie informacji o złożonym układzie
                data = {
                    'language': "pl" if self.lang_combo.currentIndex() == 0 else "en",
                    'complex_layout': 'true' if self.layout_cb.isChecked() else 'false'
                }
                headers = {"Authorization": f"Bearer {self.access_token}"}
                r = requests.post(f"{self.api_url}/process-document", files=files, data=data, headers=headers)

            if r.status_code == 200:
                self.current_task_id = r.json().get('task_id')

                ws_base = self.api_url.replace("https://", "wss://").replace("http://", "ws://")
                ws_url = f"{ws_base}/ws/status/{self.current_task_id}"

                self.websocket.open(QUrl(ws_url))
                self.status.setText("Inicjalizacja modułu wizyjnego CRNN.")
                
                # Uruchomienie ekranu ładowania
                if self.memory_game:
                    self.memory_game.close()
                self.memory_game = MemoryGameDialog(self)
                self.memory_game.show()

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
                
                # Zamknięcie okna gry, gdy zadanie dobiegnie końca
                if self.memory_game and self.memory_game.isVisible():
                    self.memory_game.accept()
                    
                self.render_results(data.get('result'))
                self.current_file_idx += 1
                self.btn_next.setEnabled(True)
                self.status.setText("Analiza gotowa. Sukces kaskady HTR!")

            elif data.get('status') == 'FAILED':
                self.websocket.close()
                
                # Zamknięcie okna w przypadku awarii chmury
                if self.memory_game and self.memory_game.isVisible():
                    self.memory_game.reject()
                    
                self.btn_next.setEnabled(True)
                self.status.setText("Wystąpił błąd podczas analizy w chmurze.")
                QMessageBox.critical(self, "Błąd", "Nie udało się przeanalizować dokumentu. Zobacz logi serwera.")

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
        # Zapisujemy wynik modelu, zanim użytkownik go zmieni
        original_text = box.current_text 
        
        # Wypełniamy okienko zapisanym wcześniej oryginalnym tekstem
        text, ok = QInputDialog.getText(self, "Korekta", "Popraw tekst:", QLineEdit.EchoMode.Normal, original_text)
        
        if ok and text:
            try:
                # Oczyszczamy teksty ze zbędnych białych znaków na brzegach
                ground_truth = text.strip()
                model_pred = original_text.strip()
                
                # Obliczenia metryk i logowanie (tylko w trybie testowym)
                if hasattr(self, 'test_mode_cb') and self.test_mode_cb.isChecked():
                    # CER
                    char_dist = Levenshtein.distance(model_pred, ground_truth)
                    self.total_char_errors += char_dist
                    self.total_chars += max(len(ground_truth), 1) # max by uniknąć dzielenia przez 0
                    
                    # WER
                    orig_words = model_pred.split()
                    gt_words = ground_truth.split()

                    # Uproszczone liczenie błędów na poziomie słów
                    word_dist = sum(1 for o, c in zip(orig_words, gt_words) if o != c) + abs(len(orig_words) - len(gt_words))
                    self.total_word_errors += word_dist
                    self.total_words += max(len(gt_words), 1)
                    
                    # Aktualizacja UI
                    cer = (self.total_char_errors / self.total_chars) * 100
                    wer = (self.total_word_errors / self.total_words) * 100
                    self.metrics_label.setText(f"Bieżąca sesja » CER: {cer:.2f}% | WER: {wer:.2f}%")

                    # Automatyczny zapis do pliku CSV
                    log_file = "test_metrics_log.csv"
                    file_exists = os.path.isfile(log_file)
                    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(["Czas", "Oryginał (Model)", "Korekta (Ground Truth)", "CER (%)", "WER (%)"])
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        writer.writerow([timestamp, model_pred, ground_truth, f"{cer:.2f}", f"{wer:.2f}"])

                box_rect = box.rect()
                
                # Zmiana struktury pod wymogi backendu
                payload = {
                    'document_id': box.document_id,
                    'word_id': box.word_id,  # Wysyłamy ID słowa
                    'correct_text': ground_truth, 
                    'box': json.dumps([box_rect.x(), box_rect.y(), box_rect.width(), box_rect.height()]) # Jako string
                }

                headers = {"Authorization": f"Bearer {self.access_token}"}
                
                # Używamy parametru 'data=' zamiast 'json='
                response = requests.post(f"{self.api_url}/submit-feedback", data=payload, headers=headers, timeout=10)
                response.raise_for_status()

                # Aktualizujemy interfejs nowym tekstem
                box.update_result(ground_truth)
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
        user_id = self.user_id_input.text()

        # Dodany user_id na końcu ścieżki
        requests.post(f"{self.api_url}/trigger-adaptation/{user_id}", headers=headers)
        QMessageBox.information(self, "Adaptacja", "Proces adaptacji wag rozpoczęty w tle.")

    def change_server_address(self):
        new_url, ok = QInputDialog.getText(
            self, "Ustawienia Sieciowe",
            "Adres serwera API (wymagane https://, np. https://api.twojafirma.pl):",
            QLineEdit.EchoMode.Normal, self.api_url
        )
        if ok and new_url:
            new_url = new_url.strip()

            # Ostrzegamy, gdy adres zaczyna się od "http://" (brak szyfrowania)
            if new_url.startswith("http://") and "localhost" not in new_url and "127.0.0.1" not in new_url:
                reply = QMessageBox.warning(
                    self, "Zagrożenie Bezpieczeństwa",
                    "Próbujesz połączyć się przez niezabezpieczony protokół HTTP (np. z publicznym IP).\nTwoje hasło może zostać przechwycone!\nCzy na pewno chcesz kontynuować?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return

            self.api_url = new_url
            save_api_url(self.api_url)
            self.status.setText(f"Adres serwera zmieniony na: {self.api_url}")
            QMessageBox.information(self, "Sukces", "Adres API został zaktualizowany.")

    def display_session_summary(self):
        """ Oblicza i wyświetla globalne statystyki sesji dla użytkownika. """
        # Zabezpieczenie przed dzieleniem przez zero, jeśli sesja się jeszcze nie zaczęła
        if self.total_chars == 0 or self.total_words == 0:
            QMessageBox.information(
                self, 
                "Statystyki Sesji", 
                "Brak zebranych danych testowych. Rozpocznij poprawianie słów w trybie testowym."
            )
            return

        # Obliczanie statystyk na podstawie zgromadzonych błędów
        cer = (self.total_char_errors / self.total_chars) * 100
        wer = (self.total_word_errors / self.total_words) * 100

        # Konstruowanie raportu statystycznego
        stats_message = (
            f"Raport Sesji\n"
            f"Suma przetworzonych znaków: {self.total_chars}\n"
            f"Suma przetworzonych słów: {self.total_words}\n"
            f"Współczynnik błędu znakowego (CER): {cer:.2f}%\n"
            f"Współczynnik błędu słownego (WER): {wer:.2f}%\n"
            f"Status: Dane zebrane z aktywnej sesji testowej."
        )

        QMessageBox.information(self, "Globalne Statystyki HTR", stats_message)

    def merge_selected_boxes(self):
        """ Umożliwia złączenie błędów segmentacji, żeby w wyniku uzyskać 1 słowo. """
        selected_items = self.scene.selectedItems()
        if len(selected_items) < 2:
            QMessageBox.warning(self, "Błąd", "Zaznacz co najmniej 2 ramki (z wciśniętym Ctrl).")
            return
            
        # Obliczamy nową ramkę obejmującą wszystkie zaznaczone
        min_x = min(item.rect().x() for item in selected_items)
        min_y = min(item.rect().y() for item in selected_items)
        max_x = max(item.rect().x() + item.rect().width() for item in selected_items)
        max_y = max(item.rect().y() + item.rect().height() for item in selected_items)
        
        doc_id = selected_items[0].document_id
        
        # Usuwamy stare ramki ze sceny
        for item in selected_items:
            self.scene.removeItem(item)
            
        # Tworzymy nową, połączoną ramkę
        new_w, new_h = max_x - min_x, max_y - min_y
        
        # Wstępna ramka (wizualizacja przed otrzymaniem wyniku)
        dummy_meta = {'word_id': 'merged', 'final_result': '...', 'confidence': 0.0}
        new_box = WordBoxItem(min_x, min_y, new_w, new_h, dummy_meta, self, doc_id)
        self.scene.addItem(new_box)
        
        self.status.setText("Trwa ponowna inferencja złączonego wycinka...")
        QApplication.processEvents() # Wymuszenie odświeżenia UI
        
        try:
            payload = {
                'document_id': doc_id,
                'box': json.dumps([min_x, min_y, new_w, new_h]),
                'language': "pl" if self.lang_combo.currentIndex() == 0 else "en"
            }
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            # Wysłanie połączonych współrzędnych z powrotem do API 
            response = requests.post(f"{self.api_url}/re-evaluate-box", data=payload, headers=headers, timeout=15)
            
            if response.status_code == 200:
                new_meta = response.json().get('meta', {})
                self.scene.removeItem(new_box)
                final_box = WordBoxItem(min_x, min_y, new_w, new_h, new_meta, self, doc_id)
                self.scene.addItem(final_box)
                self.status.setText("Zakończono ponowną inferencję.")
            else:
                self.status.setText("Błąd ponownej inferencji na serwerze.")
        except Exception as e:
            self.status.setText(f"Wystąpił błąd komunikacji: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ex = HTRMainWindow()
    ex.show()
    sys.exit(app.exec_())