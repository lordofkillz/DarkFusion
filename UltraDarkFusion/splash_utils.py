# splash_utils.py

from PyQt5.QtWidgets import QSplashScreen
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl

class SplashScreen(QSplashScreen):
    def __init__(self, gif_path):
        pixmap = QPixmap()
        super(SplashScreen, self).__init__(pixmap, Qt.WindowStaysOnTopHint)
        self.movie = QMovie(gif_path)
        self.movie.frameChanged.connect(self.set_frame)
        self.movie.start()

        self.sound_player = QMediaPlayer()
        self.sound_player.setMedia(QMediaContent(QUrl.fromLocalFile("sounds/Darkfusion.wav")))
        self.sound_player.play()

    def set_frame(self):
        self.setPixmap(self.movie.currentPixmap())

def show_splash(app):
    splash = SplashScreen("styles/gifs/darkfusion.gif")
    splash.show()
    app.processEvents()
    return splash
