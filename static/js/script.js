/**
 * 設定非表示
 */
function hideNav() {
    $('#main_nav').addClass('d-none')
    $('#btn_open').show()
}

/**
 * 設定表示
 */
function showNav() {
    $('#main_nav').removeClass('d-none')
    $('#btn_open').hide()
}

/**
 * 範囲入力値表示
 * @param {string} rngId 範囲入力ID
 * @param {string} divId 範囲表示ID
 */
function showRangeValue(rngId, divId) {
    const rng = document.querySelector('#' + rngId)
    const div = document.querySelector('#' + divId)
    div.textContent = rng.value
    rng.addEventListener('input', (event) => {
        div.textContent = event.target.value
    })
}

/**
 * 画像をdivに表示する
 * @param {[name: string]: string} testResult 結果データ
 * @param {string} img 結果データの画像ID
 * @param {string} imgId 表示するimgのid
 * @param {string} imgTitle 表示する画像のタイトル
 */
function setImage(testResult, img, imgId) {
    const b64encode = testResult[img]
    $(`#${imgId}`).prop('src', `data:image/jpeg;base64,${b64encode}`)
}

/**
 * ページボタン押下可不可設定
 */
function setPrevnextEnabled() {
    const currentIndex = Number($('#hdn_current').val())
    if (0 < currentIndex) {
        $('#btn_prev').prop('disabled', false)
        $('#btn_prev').addClass('btn-outline-primary')
        $('#btn_prev').removeClass('btn-light')
    } else {
        $('#btn_prev').prop('disabled', true)
        $('#btn_prev').removeClass('btn-outline-primary')
        $('#btn_prev').addClass('btn-light')
    }
    const testSize = Number($('#hdn_test_size').val())
    if (currentIndex < testSize - 1) {
        $('#btn_next').prop('disabled', false)
        $('#btn_next').addClass('btn-outline-primary')
        $('#btn_next').removeClass('btn-light')
    } else {
        $('#btn_next').prop('disabled', true)
        $('#btn_next').removeClass('btn-outline-primary')
        $('#btn_next').addClass('btn-light')
    }
}

/**
 * 検査結果を表示する
 * @param {JSON} testResult 検査結果
 */
function showResult(testResult) {
    $('#result').show()

    setImage(testResult, 'sample_b64encode', 'img_sample')
    setImage(testResult, 'fmap_b64encode', 'img_fmap')
    setImage(testResult, 'overlay_img_b64encode', 'img_overlay')
    setImage(testResult, 'img_hist_b64encode', 'img_hist')

    // 異常度
    const scoreString = testResult['score_string']
    const isError = testResult['is_error']
    $('#div_score').html(scoreString)
    const scoreClass = isError ? 'text-danger' : 'text-success'
    $('#div_score').removeClass('text-danger text-success')
    $('#div_score').addClass(scoreClass)

    // テスト数
    const testSize = testResult['test_size']
    $('#hdn_test_size').val(testSize)

    // ページ
    $('#div_current').html(
        `${Number($('#hdn_current').val()) + 1} / ${testSize}`
    )
    setPrevnextEnabled()
}

/**
 * フォーム送信
 * @param {event} event フォーム送信イベント
 */
function formSubmit(event) {
    try {
        event.preventDefault()

        // 入力チェック
        if ($('#file_train').get(0).files.length < 3 ||
            $('#file_test').get(0).files.length < 1) {
            alert('正常画像３枚以上と検査画像１枚以上を選択してください。')
            return
        }
        $('#index_message').hide()
        $('#result_message').show()
        $('#result_message').html('Checking or downloading dataset ...')
        $('#result').hide()
        $('#div_train').empty()
        const formData = new FormData($('#form')[0])
        $.ajax({
            url: '/predict/',
            type: 'post',
            processData: false,
            contentType: false,
            data: formData
        }).done(function (data) {
            // success
            $('#result_message').hide()
            // 正常画像
            const trainB64encodeImages = data['train_b64encode_images']
            // テスト結果
            const testResult = data['test_result']
            trainB64encodeImages.forEach((file, index) => {
                $('#div_train').append(
                    `<div class='position-relative'>
                        <i class='bi bi-arrows-angle-expand fs-4 position-absolute top-0 end-0 pe-3 i-modal' data-bs-toggle='modal' data-bs-target='#imageModal' data-bs-title='正常画像[${index + 1}]' data-bs-imgid='img_train${index}'></i>
                        <img id='img_train${index}' src='data:image/jpeg;base64,${file}' />
                    </div>`
                )
            });
            $('#hdn_current').val(0)
            showResult(testResult)
            // 値変更
            $('#hdn_value_changed').val(false)
        }).fail(function (jqXHR, textStatus, errorThrown) {
            // error
            console.log('error')
        });
    } catch (error) {
        console.error(error)
    }
}

/**
 * エラー判定変更
 */
function changeCondition() {
    try {
        const formData = new FormData($('#form')[0])
        $.ajax({
            url: '/change/',
            type: 'post',
            processData: false,
            contentType: false,
            data: formData
        }).done(function (data) {
            // success
            // テスト結果
            const testResult = data['test_result']
            showResult(testResult)
        }).fail(function (jqXHR, textStatus, errorThrown) {
            // error
            console.log('error')
        });
    } catch (error) {
        console.error(error)
    }
}

/**
 * 検査対象変更
 */
function changeConditionIndex(isUp) {
    try {
        // ページ遷移
        const currentIndex = Number($('#hdn_current').val())
        const testIndex = isUp ? currentIndex + 1 : currentIndex - 1
        const testSize = Number($('#hdn_test_size').val())
        const testIndexClip = Math.max(0, Math.min(testIndex, testSize - 1))
        $('#hdn_current').val(testIndexClip)

        const formData = new FormData($('#form')[0])
        $.ajax({
            url: `/change/${testIndexClip}`,
            type: 'post',
            processData: false,
            contentType: false,
            data: formData
        }).done(function (data) {
            // success
            // テスト結果
            const testResult = data['test_result']
            showResult(testResult)
        }).fail(function (jqXHR, textStatus, errorThrown) {
            // error
            console.log(`error:${jqXHR},${textStatus},${errorThrown}`)
        });
    } catch (error) {
        console.error(error)
    }
}

/**
 * 画像拡大表示
 */
function setModal() {
    const imageModal = document.getElementById('imageModal')
    if (imageModal) {
        imageModal.addEventListener('show.bs.modal', event => {
            const button = event.relatedTarget
            const title = button.getAttribute('data-bs-title')
            const imgid = button.getAttribute('data-bs-imgid')

            const modalTitle = imageModal.querySelector('.modal-title')
            const modalBodyImg = imageModal.querySelector('.modal-body img')

            modalTitle.textContent = title
            modalBodyImg.src = document.getElementById(imgid).src
        })
    }
}

$(function () {
    // 設定表示・非表示
    $('#btn_open').hide()
    $('#btn_close').click(hideNav)
    $('#btn_open').click(showNav)
    $(window).on('resize', function () {
        if ($('#btn_open').is(':visible')) {
            hideNav()
        }
        else {
            showNav()
        }
    })

    // サンプルデータセット使用
    $('#chk_sample').change(function () {
        $('#sel_mvtec').prop('disabled', !this.checked)
        $('#file_train').prop('disabled', this.checked)
        $('#file_test').prop('disabled', this.checked)
    })

    // 色範囲
    $('#sld_color').slider({
        range: true,
        min: 0,
        max: 255,
        values: [30, 200],
        slide: function (event, ui) {
            $('#div_color').html(`${ui.values[0]} - ${ui.values[1]}`)
            $('#hdn_color_min').val(ui.values[0])
            $('#hdn_color_max').val(ui.values[1])
        }
    });
    $('#div_color').html(`${$('#sld_color').slider('values', 0)}
            - ${$('#sld_color').slider('values', 1)}`)
    $('#hdn_color_min').val($('#sld_color').slider('values', 0))
    $('#hdn_color_max').val($('#sld_color').slider('values', 1))

    // 色範囲に相対値使用
    $('#chk_relative').change(function () {
        $('#sld_color').slider('option', 'disabled', this.checked)
    })

    // しきい値
    showRangeValue('rng_threshold', 'div_threshold')
    // スコア最小値
    showRangeValue('rng_threshold_min', 'div_threshold_min')
    // スコア最大値
    showRangeValue('rng_threshold_max', 'div_threshold_max')

    // 値変更
    $('#fld_predict_cond input').change(function () {
        $('#hdn_value_changed').val(true)
    });

    // フォーム送信
    $('#form').submit(function (event) { formSubmit(event) })

    // エラー判定変更
    $('#fld_error_cond input').change(changeCondition)

    // ページ変更
    $('#btn_prev').click(function () { changeConditionIndex(false) })
    $('#btn_next').click(function () { changeConditionIndex(true) })

    // 結果
    $('#index_message').show()
    $('#result_message').hide()
    $('#result').hide()

    // 画像拡大表示
    setModal()
})