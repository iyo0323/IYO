[【2018年】チャート/グラフ作成用JavaScriptライブラリを比較！おすすめ5選【無料】](https://www.marorika.com/entry/chart-js-library)

[JavaScriptでグラフ描画入門！全８個のライブラリをコード付きで一挙に解説！](https://paiza.hatenablog.com/entry/2016/06/07/JavaScript%E3%81%A7%E3%82%B0%E3%83%A9%E3%83%95%E6%8F%8F%E7%94%BB%E5%85%A5%E9%96%80%EF%BC%81%E5%85%A8%EF%BC%98%E5%80%8B%E3%81%AE%E3%83%A9%E3%82%A4%E3%83%96%E3%83%A9%E3%83%AA%E3%82%92%E3%82%B3%E3%83%BC)


```js
// ajaxでheaderを送る方法
function runAPI(token) {

      $.ajax( {
        type: 'get',
        url: '../searchcondition/',
        dataType: 'json',
        beforeSend: function( xhr, settings ) { xhr.setRequestHeader( 'Authorization', 'X-Authentication-Token ' + token ); },
      } )
      .done(function( data, textStatus, jqXHR ) {
        var data2 = JSON.stringify(data);
        var data3 = JSON.parse(data2);
        console.log(data2);
        alert(jqXHR.status + " " + data3);
      })
      .fail(function( data, jqXHR, textStatus, errorThrown ) {
        var data2 = JSON.stringify(data);
        var data3 = JSON.parse(data2);
        console.log(data2);
        alert(data3.status + " " + data3['responseText']);
      })
      .always(function( jqXHR, textStatus ) {
        // alert(textStatus + " " + jqXHR.status);
      });

}
```
