System.register(["__unresolved_0", "cc", "__unresolved_1"], function (_export, _context) {
  "use strict";

  var _reporterNs, _cclegacy, _decorator, Component, Node, resources, Sprite, SpriteFrame, Label, macro, Prefab, instantiate, Vec3, PhysicsSystem2D, EPhysics2DDrawFlags, director, Gloabl, _dec, _dec2, _dec3, _dec4, _dec5, _dec6, _dec7, _class, _class2, _descriptor, _descriptor2, _descriptor3, _descriptor4, _descriptor5, _descriptor6, _class3, _temp, _crd, ccclass, property, mainScene;

  function _initializerDefineProperty(target, property, descriptor, context) { if (!descriptor) return; Object.defineProperty(target, property, { enumerable: descriptor.enumerable, configurable: descriptor.configurable, writable: descriptor.writable, value: descriptor.initializer ? descriptor.initializer.call(context) : void 0 }); }

  function _defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }

  function _applyDecoratedDescriptor(target, property, decorators, descriptor, context) { var desc = {}; Object.keys(descriptor).forEach(function (key) { desc[key] = descriptor[key]; }); desc.enumerable = !!desc.enumerable; desc.configurable = !!desc.configurable; if ('value' in desc || desc.initializer) { desc.writable = true; } desc = decorators.slice().reverse().reduce(function (desc, decorator) { return decorator(target, property, desc) || desc; }, desc); if (context && desc.initializer !== void 0) { desc.value = desc.initializer ? desc.initializer.call(context) : void 0; desc.initializer = undefined; } if (desc.initializer === void 0) { Object.defineProperty(target, property, desc); desc = null; } return desc; }

  function _initializerWarningHelper(descriptor, context) { throw new Error('Decorating class property failed. Please ensure that ' + 'proposal-class-properties is enabled and runs after the decorators transform.'); }

  function _reportPossibleCrUseOfGloabl(extras) {
    _reporterNs.report("Gloabl", "./gloabl", _context.meta, extras);
  }

  return {
    setters: [function (_unresolved_) {
      _reporterNs = _unresolved_;
    }, function (_cc) {
      _cclegacy = _cc.cclegacy;
      _decorator = _cc._decorator;
      Component = _cc.Component;
      Node = _cc.Node;
      resources = _cc.resources;
      Sprite = _cc.Sprite;
      SpriteFrame = _cc.SpriteFrame;
      Label = _cc.Label;
      macro = _cc.macro;
      Prefab = _cc.Prefab;
      instantiate = _cc.instantiate;
      Vec3 = _cc.Vec3;
      PhysicsSystem2D = _cc.PhysicsSystem2D;
      EPhysics2DDrawFlags = _cc.EPhysics2DDrawFlags;
      director = _cc.director;
    }, function (_unresolved_2) {
      Gloabl = _unresolved_2.default;
    }],
    execute: function () {
      _crd = true;

      _cclegacy._RF.push({}, "e375bBQg81OB4IWoOJs6ntq", "mainScene", undefined);

      ({
        ccclass,
        property
      } = _decorator);
      /**
       * Predefined variables
       * Name = mainScene
       * DateTime = Mon Feb 14 2022 11:08:04 GMT+0800 (中国标准时间)
       * Author = tc123456
       * FileBasename = mainScene.ts
       * FileBasenameNoExtension = mainScene
       * URL = db://assets/typescript/mainScene.ts
       * ManualUrl = https://docs.cocos.com/creator/3.4/manual/zh/
       *
       */

      _export("mainScene", mainScene = (_dec = ccclass('mainScene'), _dec2 = property({
        type: Prefab
      }), _dec3 = property({
        type: Prefab
      }), _dec4 = property({
        type: Prefab
      }), _dec5 = property({
        type: Node
      }), _dec6 = property({
        type: Node
      }), _dec7 = property({
        type: Node
      }), _dec(_class = (_class2 = (_temp = _class3 = class mainScene extends Component {
        constructor() {
          super(...arguments);

          _defineProperty(this, "exp", 0);

          _defineProperty(this, "btnSoldier", null);

          _defineProperty(this, "btnHZ", null);

          _defineProperty(this, "btnLB", null);

          _defineProperty(this, "coinLabel", null);

          _defineProperty(this, "expLabel", null);

          _initializerDefineProperty(this, "soldier", _descriptor, this);

          _initializerDefineProperty(this, "heroHz", _descriptor2, this);

          _initializerDefineProperty(this, "heroGy", _descriptor3, this);

          _defineProperty(this, "jz", null);

          _initializerDefineProperty(this, "winNode", _descriptor4, this);

          _initializerDefineProperty(this, "btn_close", _descriptor5, this);

          _initializerDefineProperty(this, "kj", _descriptor6, this);

          _defineProperty(this, "arr", []);
        }

        static get Instance() {
          if (!this.instance) {
            this.instance = new mainScene();
          }

          return this.instance;
        }

        onLoad() {
          mainScene.instance = this;
          PhysicsSystem2D.instance.enable = true;
          PhysicsSystem2D.instance.debugDrawFlags = EPhysics2DDrawFlags.Aabb | EPhysics2DDrawFlags.Pair | EPhysics2DDrawFlags.CenterOfMass | EPhysics2DDrawFlags.Joint | EPhysics2DDrawFlags.Shape;
          this.coinLabel = this.node.getChildByName("dikuan").getChildByName("Sprite2").getChildByName("num1").getComponent(Label);
          this.btnSoldier = this.node.getChildByName("dikuan").getChildByName("Layout2").getChildByName("sz-0");
          this.btnHZ = this.node.getChildByName("dikuan").getChildByName("Layout2").getChildByName("gj-1");
          this.btnLB = this.node.getChildByName("dikuan").getChildByName("Layout2").getChildByName("zd-1");
          this.expLabel = this.node.getChildByName("dikuan").getChildByName("Sprite2").getChildByName("num2").getComponent(Label);
          this.jz = this.node.getChildByName("jz");
          this.kj = this.node.getChildByName("kj");
        }

        start() {
          //this.solvingUI.active=false;
          //const audioSource = this.node.getComponent(AudioSource)!;
          //this._audioSource = audioSource;
          // let data = sys.localStorage.getItem('userda');
          // if(data){
          //     this.onBtnClose();
          //     return;
          // }
          // sys.localStorage.setItem('userda', JSON.stringify("userda"));
          this.chooseHero();
          this.winNode.getChildByName("btn").on(Node.EventType.MOUSE_DOWN, touch => {
            console.log('Mouse down');
            this.onBtnClose();
          }, this);
          this.kj.on(Node.EventType.MOUSE_DOWN, touch => {
            console.log('Mouse down');
            mainScene.js = true;
          }, this);
          this.btn_close.on(Node.EventType.MOUSE_DOWN, touch => {
            console.log('Mouse down');
            this.onBtnClose();
          }, this);
          this.btnSoldier.on(Node.EventType.MOUSE_DOWN, touch => {
            console.log('Mouse down');

            if (this.btnSoldier.getComponent(Sprite).spriteFrame.name != 'xb-1') {
              console.log("小兵");

              if ((_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
                error: Error()
              }), Gloabl) : Gloabl).coin >= 5) {
                (_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
                  error: Error()
                }), Gloabl) : Gloabl).coin -= 5;
              }

              this.chooseHero();
              this.coinLabel.string = ": " + (_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
                error: Error()
              }), Gloabl) : Gloabl).coin.toString();
              this.createHero(0);
            }
          }, this);
          this.btnHZ.on(Node.EventType.MOUSE_DOWN, touch => {
            console.log('Mouse down');
            var sp = this.btnHZ.getComponent(Sprite).spriteFrame;

            if (this.btnHZ.getComponent(Sprite).spriteFrame.name != 'hz-1') {
              console.log("黄忠");

              if ((_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
                error: Error()
              }), Gloabl) : Gloabl).coin >= 10) {
                (_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
                  error: Error()
                }), Gloabl) : Gloabl).coin -= 10;
              }

              this.chooseHero();
              this.coinLabel.string = ": " + (_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
                error: Error()
              }), Gloabl) : Gloabl).coin.toString();
              this.createHero(1);
            }
          }, this);
          this.btnLB.on(Node.EventType.MOUSE_DOWN, touch => {
            console.log('Mouse down');

            if (this.btnLB.getComponent(Sprite).spriteFrame.name != 'gy-1') {
              console.log("吕布");

              if ((_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
                error: Error()
              }), Gloabl) : Gloabl).coin >= 30) {
                (_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
                  error: Error()
                }), Gloabl) : Gloabl).coin -= 30;
              }

              this.chooseHero();
              this.coinLabel.string = ": " + (_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
                error: Error()
              }), Gloabl) : Gloabl).coin.toString();
              this.createHero(2);
            }
          }, this);
          this.coinLabel.string = ": " + (_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
            error: Error()
          }), Gloabl) : Gloabl).coin.toString();
          this.expLabel.string = ": " + this.exp.toString();
          this.createSoldier();
        }

        onBtnClose() {
          this.node.removeFromParent();
          director.getScene().removeFromParent(); //this._audioSource.stop();

          var queryString = window.location.search;
          console.log("当前地址" + queryString);
          var urlParams = new URLSearchParams(queryString);
          var code = this.getQueryVariable('url');
          console.log("截取的url" + code);
          window.location.href = code;
        }

        getQueryVariable(variable) {
          var search = "http://pjax.weisuiyu.com/?url=http://www.baidu.com";
          var query = window.location.search.substring(1);
          var vars = query.split("&");

          for (var i = 0; i < vars.length; i++) {
            var pair = vars[i].split("=");
            return pair[1];
          }
        }

        chooseHero() {
          var url1 = "";
          var url2 = "";
          var url3 = "";

          if ((_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
            error: Error()
          }), Gloabl) : Gloabl).coin >= 5) {
            url1 = "xb-0/spriteFrame";
          } else {
            url1 = "xb-1/spriteFrame";
          }

          if ((_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
            error: Error()
          }), Gloabl) : Gloabl).coin >= 10) {
            url2 = "hz-0/spriteFrame";
          } else {
            url2 = "hz-1/spriteFrame";
          }

          if ((_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
            error: Error()
          }), Gloabl) : Gloabl).coin >= 30) {
            url3 = "gy-0/spriteFrame";
          } else {
            url3 = "gy-1/spriteFrame";
          }

          resources.load(url1, SpriteFrame, (err, spriteFrame) => {
            this.btnSoldier.getComponent(Sprite).spriteFrame = spriteFrame;
          });
          resources.load(url2, SpriteFrame, (err, spriteFrame) => {
            this.btnHZ.getComponent(Sprite).spriteFrame = spriteFrame;
          });
          resources.load(url3, SpriteFrame, (err, spriteFrame) => {
            this.btnLB.getComponent(Sprite).spriteFrame = spriteFrame;
          });
        }

        createSoldier() {
          var interval = 2; // 开始延时

          var delay = 1;
          var array = [0, 0, 0, 0, 1, 0, 0, 1, 0, 2];
          var index = 0;
          this.schedule(function () {
            // 这里的 this 指向 component
            var prefab = null;

            switch (array[index]) {
              case 0:
                prefab = this.soldier;
                break;

              case 1:
                prefab = this.heroHz;
                break;

              case 2:
                prefab = this.heroGy;
                break;

              default:
                break;
            }

            if ((_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
              error: Error()
            }), Gloabl) : Gloabl).speed > 0 && index <= 9) {
              var item = instantiate(prefab);
              item.name = "enemy";
              item.setPosition(new Vec3(1100, 0, 0));
              this.jz.addChild(item);

              if (array[index] == 0) {
                item.setScale(1, 1, 1);
              } else {
                item.setScale(-1, 1, 1);
              }

              index += 1;
            }
          }, interval, macro.REPEAT_FOREVER, delay);
        }

        createHero(index) {
          // 这里的 this 指向 component
          var prefab = null;

          switch (index) {
            case 0:
              prefab = this.soldier;
              break;

            case 1:
              prefab = this.heroHz;
              break;

            case 2:
              prefab = this.heroGy;
              break;

            default:
              break;
          }

          var item = instantiate(prefab);
          item.name = "hero";

          if (index == 0) {
            item.setScale(-1, 1, 1);
          } else {
            item.setScale(1, 1, 1);
          }

          item.setPosition(new Vec3(200, 0, 0));
          this.jz.addChild(item);
        }

        refresh() {
          this.exp += 10;
          this.coinLabel.string = ": " + (_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
            error: Error()
          }), Gloabl) : Gloabl).coin.toString();
          this.expLabel.string = ": " + this.exp.toString();
          this.chooseHero();
        }

      }, _defineProperty(_class3, "instance", void 0), _defineProperty(_class3, "js", false), _temp), (_descriptor = _applyDecoratedDescriptor(_class2.prototype, "soldier", [_dec2], {
        configurable: true,
        enumerable: true,
        writable: true,
        initializer: function initializer() {
          return null;
        }
      }), _descriptor2 = _applyDecoratedDescriptor(_class2.prototype, "heroHz", [_dec3], {
        configurable: true,
        enumerable: true,
        writable: true,
        initializer: function initializer() {
          return null;
        }
      }), _descriptor3 = _applyDecoratedDescriptor(_class2.prototype, "heroGy", [_dec4], {
        configurable: true,
        enumerable: true,
        writable: true,
        initializer: function initializer() {
          return null;
        }
      }), _descriptor4 = _applyDecoratedDescriptor(_class2.prototype, "winNode", [_dec5], {
        configurable: true,
        enumerable: true,
        writable: true,
        initializer: function initializer() {
          return null;
        }
      }), _descriptor5 = _applyDecoratedDescriptor(_class2.prototype, "btn_close", [_dec6], {
        configurable: true,
        enumerable: true,
        writable: true,
        initializer: function initializer() {
          return null;
        }
      }), _descriptor6 = _applyDecoratedDescriptor(_class2.prototype, "kj", [_dec7], {
        configurable: true,
        enumerable: true,
        writable: true,
        initializer: function initializer() {
          return null;
        }
      })), _class2)) || _class));
      /**
       * [1] Class member could be defined like this.
       * [2] Use `property` decorator if your want the member to be serializable.
       * [3] Your initialization goes here.
       * [4] Your update function goes here.
       *
       * Learn more about scripting: https://docs.cocos.com/creator/3.4/manual/zh/scripting/
       * Learn more about CCClass: https://docs.cocos.com/creator/3.4/manual/zh/scripting/ccclass.html
       * Learn more about life-cycle callbacks: https://docs.cocos.com/creator/3.4/manual/zh/scripting/life-cycle-callbacks.html
       */


      _cclegacy._RF.pop();

      _crd = false;
    }
  };
});
//# sourceMappingURL=mainScene.js.map