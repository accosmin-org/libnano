#include <QLabel>
#include <QAction>
#include <QPainter>
#include <QMenuBar>
#include <QComboBox>
#include <QKeyEvent>
#include <QStatusBar>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPushButton>
#include "qt5mainwindow.h"

ImageView::ImageView(QWidget* parent, Qt::WindowFlags flags) :
    QWidget(parent, flags)
{
    setMinimumSize(400, 300);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

void ImageView::dataset(ImagesDataset* dataset)
{
    m_index = 0;
    m_zoom2 = 0;
    m_dataset = dataset;
    update();
}

void ImageView::nextImage()
{
    if (m_dataset != nullptr)
    {
        m_index = std::min(m_index + 1, m_dataset->samples() - 1);
        update();
    }
}

void ImageView::prevImage()
{
    if (m_dataset != nullptr)
    {
        m_index = std::max(m_index - 1, nano::tensor_size_t(0));
        update();
    }
}

void ImageView::zoomOut()
{
    if (m_dataset != nullptr && m_zoom2 > -3)
    {
        -- m_zoom2;
        update();
    }
}

void ImageView::zoomIn()
{
    if (m_dataset != nullptr && m_zoom2 < 3)
    {
        ++ m_zoom2;
        update();
    }
}

qreal ImageView::zoom(const int pixels) const
{
    return  (m_zoom2 >= 0) ?
            (qreal(pixels) * qreal(1U << uint32_t(+m_zoom2))) :
            (qreal(pixels) / qreal(1U << uint32_t(-m_zoom2)));
}

QImage ImageView::image(const nano::tensor_size_t index) const
{
    assert(m_dataset != nullptr);
    assert(index >= 0 && index < m_dataset->samples());

    const auto image = m_dataset->input(index);
    const auto width = image.size<0>();
    const auto height = image.size<1>();
    const auto channels = image.size<2>();
    assert(channels == 1 || channels == 3);

    const auto format = channels == 3 ? QImage::Format_RGB888 : QImage::Format_Grayscale8;
    const auto stride = channels * width;

    return QImage(image.data(), width, height, stride, format);
}

QString ImageView::label(const nano::tensor_size_t index) const
{
    assert(m_dataset != nullptr);
    assert(index >= 0 && index < m_dataset->samples());

    const auto tfeature = m_dataset->tfeature();
    if (tfeature.discrete())
    {
        const auto target = m_dataset->target(index);

        QString label;
        for (nano::tensor_size_t ilabel = 0; ilabel < target.size(); ++ ilabel)
        {
            if (nano::is_pos_target(target(ilabel)))
            {
                assert(static_cast<size_t>(ilabel) < tfeature.labels().size());
                if (!label.isEmpty())
                {
                    label += ",";
                }
                label += QString::fromStdString(tfeature.labels()[static_cast<size_t>(ilabel)]);
            }
        }

        return label;
    }
    else
    {
        return "";
    }
}

QRectF ImageView::center(const QRectF& area, const qreal width, const qreal height)
{
    return {area.left() + (area.width() - width) / 2, area.top() + (area.height() - height) / 2, width, height};
}

void ImageView::paintEvent(QPaintEvent*)
{
    if (m_dataset == nullptr)
    {
        return;
    }

    assert(m_index >= 0 && m_index < m_dataset->samples());

    QPainter painter(this);

    const auto border = 4.0;
    const auto thumbsize = 32.0;
    const auto fontheight = QFontMetrics(painter.font()).height() * 1.2;

    const auto viewRect = rect();

    const auto labelRect = QRectF(
        viewRect.left() + border,
        viewRect.top() + border,
        viewRect.width() - 2 * border,
        fontheight);

    const auto imageRect = QRectF(
        viewRect.left() + border,
        viewRect.top() + fontheight + 2 * border,
        viewRect.width() - 2 * border,
        viewRect.height() - 3 * border - thumbsize - fontheight);

    const auto thumbRect = QRectF(
        viewRect.left() + border,
        viewRect.bottom() - border - thumbsize,
        viewRect.width() - 2 * border,
        thumbsize);

    const auto image = this->image(m_index);
    painter.drawImage(
        center(imageRect, zoom(image.width()), zoom(image.height())),
        image);

    const auto label = this->label(m_index);
    QFont font = painter.font();
    font.setFamily("Courier New");
    painter.setFont(font);
    painter.drawText(labelRect, Qt::AlignCenter, label);

    const auto middleThumbRect = center(thumbRect, thumbsize, thumbsize);
    painter.drawImage(middleThumbRect, image);

    painter.setBrush(Qt::NoBrush);
    painter.setPen(QPen(QColor(255, 0, 0), border / 2));
    painter.drawRect(QRectF(
        middleThumbRect.left() - border / 2, middleThumbRect.top() - border / 2,
        thumbsize + border, thumbsize + border));

    auto left = middleThumbRect.left() - thumbsize - border;
    for (auto index = m_index - 1; index >= 0 && left >= thumbRect.left(); -- index, left -= thumbsize + border) // NOLINT(cert-flp30-c)
    {
        painter.drawImage(QRectF(left, middleThumbRect.top(), thumbsize, thumbsize), this->image(index));
    }

    left = middleThumbRect.right() + border;
    for (auto index = m_index + 1; index < m_dataset->samples() && left + thumbsize <= thumbRect.right();
        ++ index, left += thumbsize + border) // NOLINT(cert-flp30-c)
    {
        painter.drawImage(QRectF(left, middleThumbRect.top(), thumbsize, thumbsize), this->image(index));
    }

    // TODO: option to move the index to a given position
    // TODO: filter by fold and/or label
    // TODO: dataset loading in a separate thread
    // TODO: widget to display logging cout/cerr from libnano
}

void ImageView::keyPressEvent(QKeyEvent* event)
{
    switch (event->key())
    {
    case Qt::Key_Left:      prevImage(); break;
    case Qt::Key_Right:     nextImage(); break;
    case Qt::Key_Minus:     zoomOut(); break;
    case Qt::Key_Plus:      zoomIn(); break;
    default:                break;
    }
}

ImageWidget::ImageWidget(QWidget* parent, Qt::WindowFlags flags) :
    QWidget(parent, flags),
    m_dataLabel(new QLabel()),
    m_dataCombo(new QComboBox()),
    m_loadButton(new QPushButton(tr("Load"))),
    m_imageView(new ImageView())
{
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    for (const auto& id : nano::imclass_dataset_t::all().ids())
    {
        m_dataCombo->addItem(QString::fromStdString(id));
    }

    auto dataLayout = new QHBoxLayout();
    dataLayout->addWidget(m_dataCombo);
    dataLayout->addWidget(m_loadButton);

    auto ctrlLayout = new QVBoxLayout();
    ctrlLayout->addLayout(dataLayout);
    ctrlLayout->addWidget(m_dataLabel);
    ctrlLayout->addStretch();

    auto mainLayout = new QHBoxLayout();
    mainLayout->addLayout(ctrlLayout);
    mainLayout->addWidget(m_imageView);
    setLayout(mainLayout);

    connect(m_loadButton, &QPushButton::clicked, this, &ImageWidget::load);
}

void ImageWidget::load()
{
    const auto datasetName = m_dataCombo->currentText();

    m_dataset = nano::imclass_dataset_t::all().get(datasetName.toStdString());
    assert(m_dataset);
    m_dataset->load();

    assert(m_dataset->folds() > 0);

    QString label;
    label += tr("name........%1\n").arg(datasetName);
    label += tr("|-folds.....%1\n").arg(m_dataset->folds());
    label += tr("|-samples...%1\n").arg(m_dataset->samples());
    label += tr("  |-train...%1\n").arg(m_dataset->samples({size_t{0}, nano::protocol::train}));
    label += tr("  |-valid...%1\n").arg(m_dataset->samples({size_t{0}, nano::protocol::valid}));
    label += tr("  |-test....%1\n").arg(m_dataset->samples({size_t{0}, nano::protocol::test}));
    label += tr("|-inputs....%1\n").arg(QString::fromStdString(nano::to_string(m_dataset->idim())));
    label += tr("|-targets...%1 (%4)\n").arg(QString::fromStdString(nano::to_string(m_dataset->tdim())))
        .arg(m_dataset->tfeature().discrete() ? "discrete" : "continuous");

    QFont font = m_dataLabel->font();
    font.setFamily("Courier New");
    m_dataLabel->setFont(font);
    m_dataLabel->setText(label);

    m_imageView->dataset(m_dataset.get());
    m_imageView->setFocus(Qt::OtherFocusReason);
}

MainWindow::MainWindow(QWidget* parent, Qt::WindowFlags flags) :
    QMainWindow(parent, flags),
    m_imageWidget(new ImageWidget())
{
    setCentralWidget(m_imageWidget);

    makeMenu();
    makeStatusBar();
}

void MainWindow::makeStatusBar()
{
    statusBar()->showMessage(tr("Ready"));
}

void MainWindow::makeMenu()
{
    auto viewMenu = menuBar()->addMenu(tr("&View"));
    viewMenu->addAction(tr("&Previous Image"), this, [=] () { m_imageWidget->prevImage(); }, QKeySequence(tr("Left")));
    viewMenu->addAction(tr("&Next Image"), this, [=] () { m_imageWidget->nextImage(); }, QKeySequence(tr("Right")));
    viewMenu->addSeparator();
    viewMenu->addAction(tr("Zoom &In"), this, [=] () { m_imageWidget->zoomIn(); }, QKeySequence::ZoomIn);
    viewMenu->addAction(tr("Zoom &Out"), this, [=] () { m_imageWidget->zoomOut(); }, QKeySequence::ZoomOut);
}
