Install
=================================

Create New Project (Version=5.5, ProjectName=hoge)
```sh
$ composer create-project --prefer-dist laravel/laravel=5.5.* hoge
```

vendorのInstall
```sh
$ composer install
```

Specified key was too long error
```sh
https://laravel-news.com/laravel-5-4-key-too-long-error
```

Install Auth Func
```sh
$ php artisan make:auth
$ php artisan migrate
$ php artisan key:generate
$ php artisan config:clear
```

Install Mail Reset Password
```sh
$ php artisan make:notification MailResetPasswordToken
```

Install DebugBar
```sh
$ composer require barryvdh/laravel-debugbar --dev
```


Create Files
=================================

Create Migration File (Create)
```sh
$ php artisan make:migration create_tweets_table --create=tweets
```

Create Table File (Add Column)
```sh
$ php artisan make:migration add_hash_tag_to_tweet_table --table=tweets
```

Create Controller File
```sh
$ php artisan make:controller TweetController
```

Create Model File (ModelName=Tweet)
```sh
$ php artisan make:model Tweet
```

Create Seed (Data)
```sh
$ php artisan make:seeder TestSeeder
```


Migration (Run & Rollback)
=================================

Do Migration
```sh
$ php artisan migrate
```

Rollback the last 5 migrations
```sh
$ php artisan migrate:rollback --step=5
```

Rollback all migrations
```sh
$ php artisan migrate:reset
```

Rollback the last 5 migrations, and do migration again
```sh
$ php artisan migrate:refresh --step=5
```

Run Specific Seed
```sh
$ php artisan db:seed --class=Sample001DataSeeder
$ php artisan db:seed --class=Sample180328DataSeeder
```


Reload
=================================

Reload Files
```sh
$ cd C:\xampp\htdocs\{projectname}
$ composer dump-autoload
```

Refresh cache
```sh
php artisan config:clear
php artisan view:clear
composer dump-autoload
php artisan clear-compiled
php artisan optimize
php artisan config:cache
```


Settings
=================================

.envの設定

MariaDBの設定例
```php
DB_CONNECTION=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_DATABASE={dbname}
DB_USERNAME=root
DB_PASSWORD=
```

PostgreSQLの設定例
```php
DB_CONNECTION=pgsql
DB_HOST=127.0.0.1
DB_PORT=5432
DB_DATABASE={dbname}
DB_USERNAME=postgres
DB_PASSWORD=postgres
```

SMTPの設定例
```php
MAIL_DRIVER=smtp
MAIL_HOST=smtp.mailtrap.io
MAIL_PORT=2525
MAIL_USERNAME=null
MAIL_PASSWORD=null
MAIL_ENCRYPTION=null
```


Code
=================================


RegisterController
```php
public function showRegistrationForm()
{
	abort(404);
}
```


\app\Http\Controllers\Apis\CsvMaker.php
```php
<?php 
namespace App\Http\Controllers;
class CsvMaker extends Controller
{    
    public function __construct()
    {
        
    }

    /**
     * CSVダウンロード
     * @param array $list
     * @param array $header
     * @param string $filename
     * @return \Illuminate\Http\Response
     */
    public static function download($list, $header, $filename)
    {
        if (count($header) > 0) {
            // $listの先頭に$headerを追加する
            array_unshift($list, $header);
        }
        
        // 読み込み／書き出し用にオープンします。 ファイルポインタをファイルの先頭に置きます。
        $stream = fopen('php://temp', 'r+b');
        foreach ($list as $row) {
            // 行を CSV 形式にフォーマットし、ファイルポインタに書き込む
            fputcsv($stream, $row);
        }
        
        // ファイルポインタの位置を先頭に戻す
        rewind($stream);
        // 残りのストリームを文字列に読み込んで、replaceする
        $csv = str_replace(PHP_EOL, "\r\n", stream_get_contents($stream));
        // 文字コードをSJIS-winに変換する
        $csv = mb_convert_encoding($csv, 'SJIS-win', 'auto');
        
        // return用の$headersを生成
        $headers = array(
            'Content-Type' => 'text/csv',
            'Content-Disposition' => "attachment; filename=$filename",
        );
        
        // 結果を戻す
        return \Response::make($csv, 200, $headers);
    }
}
```

validate
```php
//漢字・ひらがな・カタカナ
public function validateKanji($attribute, $value)
{
	if (preg_match('/^(\p{Han}|\p{Hiragana}|\p{Katakana})+$/u', $value)) {
		return true;
	}
	return false;
}
```


[Laravel複数項目のValidation](https://nextat.co.jp/staff/archives/126)
```php
// 電話番号の桁数チェック（True: 10桁以上, False: 9桁以下）
public function validateTelLength10($attribute, $value, $params, $validator) {
	$data = $validator->getData();
	$len = strlen($data['tel_1']) + strlen($data['tel_2']) + strlen($data['tel_3']);
	if( $len >= 10){
		return true;
	}
	return false;
}

'tel_3' => ['required', 'regex:/^[0-9]{1,4}$/', 'tellength10'],

'tel_3.tellength10' => '電話番号を確認してください。',
```


Others
=================================

[日本語化（Laravel5.5）](https://qiita.com/Takahisa1984/items/f2d4347031adbf645594)

[Htmlable](https://qiita.com/horikeso/items/f891ea52e2fcda89d170)

```php
// C:\xampp\htdocs\nh_jizensodan\src\vendor\laravel\framework\src\Illuminate\Support\helpers.php
AppServiceProvider　→　Blade::doubleEncode();
```

[LaravelのGate(ゲート)機能で権限(ロール)によるアクセス制限を実装する](https://www.ritolab.com/entry/56)

[Laravel でasset()やurl()が返すURLを『https』 にするためのメモ](http://fushigi.hatenadiary.com/entry/2018/04/12/223137)







Startup Laravel Server
```sh
php artisan serve
```

Update Vender?
```sh
$ php composer.phar self-update
$ php composer.phar dump-auto
```

```sh
php artisan vendor:publish
```
