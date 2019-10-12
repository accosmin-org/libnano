#pragma once

#include <QWidget>
#include <QMainWindow>
#include <nano/tensor/index.h>
#include <nano/imclass.h>

using ImagesDataset = nano::imclass_dataset_t;
using UPImagesDataset = nano::rimclass_dataset_t;

class QLabel;
class QComboBox;
class QPushButton;

class ImageView final : public QWidget
{
    Q_OBJECT

public:

    explicit ImageView(QWidget* parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());

    void zoomIn();
    void zoomOut();
    void nextImage();
    void prevImage();
    void dataset(ImagesDataset*);
    void paintEvent(QPaintEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;

private:

    qreal zoom(int pixels) const;
    QImage image(nano::tensor_size_t index) const;
    QString label(nano::tensor_size_t index) const;
    static QRectF center(const QRectF& area, qreal width, qreal height);

    // attributes
    nano::tensor_size_t m_index{0};                 ///<
    int                 m_zoom2{0};                 ///<
    ImagesDataset*      m_dataset{nullptr};         ///<
};

class ImageWidget final : public QWidget
{
    Q_OBJECT

public:

    explicit ImageWidget(QWidget* parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());

public slots:

    void load();
    void zoomIn() { m_imageView->zoomIn(); }
    void zoomOut() { m_imageView->zoomOut(); }
    void nextImage() { m_imageView->nextImage(); }
    void prevImage() { m_imageView->prevImage(); }

private:

    // attributes
    QLabel*             m_dataLabel{nullptr};       ///<
    QComboBox*          m_dataCombo{nullptr};       ///<
    QPushButton*        m_loadButton{nullptr};      ///<
    ImageView*          m_imageView{nullptr};       ///<
    UPImagesDataset     m_dataset;                  ///<
};

class MainWindow final : public QMainWindow
{
    Q_OBJECT

public:

    explicit MainWindow(QWidget* parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags());

private:

    void makeMenu();
    void makeStatusBar();

    // attributes
    ImageWidget*    m_imageWidget{nullptr};     ///<
};
