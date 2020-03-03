===============================================
Mermaid Sample
===============================================
.. mermaid::

    sequenceDiagram

        participant cli as AAA
        participant svr as BBB
        participant que as CCC
        participant bat as DDD

        cli ->>+ svr :X処理

        note right of svr : X検証

        svr -->>- cli :X返却

        cli ->>+ svr :Y処理

        svr -->>+ que :Y依頼登録
        
        que ->>+ bat :Y実行依頼
        
        svr -->>- cli :Y返却

        loop Y完了するまで

            cli ->>+ svr :Z確認
            bat -->> bat :Y処理
            note right of bat : YYY<br>YYY<br>YYY
            bat -->> svr :Y実行結果
            svr -->>- cli :Z結果報告
        
        end
        bat -->>- que :Y通知

        cli ->>+ svr :W取得
        svr -->>- cli :W返却
