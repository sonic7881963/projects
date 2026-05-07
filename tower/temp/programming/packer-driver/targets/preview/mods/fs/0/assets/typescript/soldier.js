System.register(["__unresolved_0", "cc", "__unresolved_1", "__unresolved_2"], function (_export, _context) {
  "use strict";

  var _reporterNs, _cclegacy, _decorator, Component, Contact2DType, BoxCollider2D, PhysicsSystem2D, Animation, Sprite, Gloabl, mainScene, _dec, _class, _class2, _descriptor, _descriptor2, _temp, _crd, ccclass, property, soldier;

  function _initializerDefineProperty(target, property, descriptor, context) { if (!descriptor) return; Object.defineProperty(target, property, { enumerable: descriptor.enumerable, configurable: descriptor.configurable, writable: descriptor.writable, value: descriptor.initializer ? descriptor.initializer.call(context) : void 0 }); }

  function _defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }

  function _applyDecoratedDescriptor(target, property, decorators, descriptor, context) { var desc = {}; Object.keys(descriptor).forEach(function (key) { desc[key] = descriptor[key]; }); desc.enumerable = !!desc.enumerable; desc.configurable = !!desc.configurable; if ('value' in desc || desc.initializer) { desc.writable = true; } desc = decorators.slice().reverse().reduce(function (desc, decorator) { return decorator(target, property, desc) || desc; }, desc); if (context && desc.initializer !== void 0) { desc.value = desc.initializer ? desc.initializer.call(context) : void 0; desc.initializer = undefined; } if (desc.initializer === void 0) { Object.defineProperty(target, property, desc); desc = null; } return desc; }

  function _initializerWarningHelper(descriptor, context) { throw new Error('Decorating class property failed. Please ensure that ' + 'proposal-class-properties is enabled and runs after the decorators transform.'); }

  function _reportPossibleCrUseOfGloabl(extras) {
    _reporterNs.report("Gloabl", "./gloabl", _context.meta, extras);
  }

  function _reportPossibleCrUseOfmainScene(extras) {
    _reporterNs.report("mainScene", "./mainScene", _context.meta, extras);
  }

  return {
    setters: [function (_unresolved_) {
      _reporterNs = _unresolved_;
    }, function (_cc) {
      _cclegacy = _cc.cclegacy;
      _decorator = _cc._decorator;
      Component = _cc.Component;
      Contact2DType = _cc.Contact2DType;
      BoxCollider2D = _cc.BoxCollider2D;
      PhysicsSystem2D = _cc.PhysicsSystem2D;
      Animation = _cc.Animation;
      Sprite = _cc.Sprite;
    }, function (_unresolved_2) {
      Gloabl = _unresolved_2.default;
    }, function (_unresolved_3) {
      mainScene = _unresolved_3.mainScene;
    }],
    execute: function () {
      _crd = true;

      _cclegacy._RF.push({}, "a4916AjCGhM9JVQvvXdlJfc", "soldier", undefined);

      ({
        ccclass,
        property
      } = _decorator);

      _export("soldier", soldier = (_dec = ccclass('soldier'), _dec(_class = (_class2 = (_temp = class soldier extends Component {
        constructor() {
          super(...arguments);

          _initializerDefineProperty(this, "hp", _descriptor, this);

          _initializerDefineProperty(this, "fight", _descriptor2, this);

          _defineProperty(this, "selfCollider", void 0);

          _defineProperty(this, "otherCollider", void 0);

          _defineProperty(this, "speed", 100);

          _defineProperty(this, "num", 0);

          _defineProperty(this, "war", false);
        }

        onLoad() {
          var box = this.node.getComponent(BoxCollider2D);

          if (this.node.getComponent(Sprite).spriteFrame.name == "soldier") {
            this.hp = 20;
            this.fight = 5;
            box.tag = 1;
          } else if (this.node.getComponent(Sprite).spriteFrame.name == "heroHz") {
            this.hp = 5;
            this.fight = 10;
            box.tag = 2;
          } else if (this.node.getComponent(Sprite).spriteFrame.name == "heroGy") {
            this.hp = 30;
            this.fight = 20;
            box.tag = 3;
          }
        }

        start() {
          // [3]
          var collider = this.node.getComponent(BoxCollider2D);

          if (collider) {
            var box = this.node.getComponent(BoxCollider2D);

            if (this.node.name == "hero") {
              box.group = 2;
            } else {
              box.group = 4;
            }

            collider.on(Contact2DType.BEGIN_CONTACT, this.onBeginContact, this);

            if (PhysicsSystem2D.instance) {
              PhysicsSystem2D.instance.on(Contact2DType.BEGIN_CONTACT, this.onBeginContact, this);
            }
          }
        }

        update(deltaTime) {
          if (this.node.name == "hero") {
            if (this.hp > 0) {
              if (this.speed) {
                if ((_crd && mainScene === void 0 ? (_reportPossibleCrUseOfmainScene({
                  error: Error()
                }), mainScene) : mainScene).js == true && this.war != true) {
                  this.speed = 150;
                }

                this.node.setPosition(this.node.position.x + this.speed * deltaTime, this.node.position.y);
              } else {
                if ((_crd && mainScene === void 0 ? (_reportPossibleCrUseOfmainScene({
                  error: Error()
                }), mainScene) : mainScene).js == true && this.war != true) {
                  (_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
                    error: Error()
                  }), Gloabl) : Gloabl).speed = 150;
                }

                this.node.setPosition(this.node.position.x + (_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
                  error: Error()
                }), Gloabl) : Gloabl).speed * deltaTime, this.node.position.y);
              }
            }
          } else {
            if (this.hp > 0) {
              if (this.speed) {
                if ((_crd && mainScene === void 0 ? (_reportPossibleCrUseOfmainScene({
                  error: Error()
                }), mainScene) : mainScene).js == true && this.war != true) {
                  this.speed = 150;
                }

                this.node.setPosition(this.node.position.x - this.speed * deltaTime, this.node.position.y);
              } else {
                if ((_crd && mainScene === void 0 ? (_reportPossibleCrUseOfmainScene({
                  error: Error()
                }), mainScene) : mainScene).js == true && this.war != true) {
                  (_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
                    error: Error()
                  }), Gloabl) : Gloabl).speed = 150;
                }

                this.node.setPosition(this.node.position.x - (_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
                  error: Error()
                }), Gloabl) : Gloabl).speed * deltaTime, this.node.position.y);
              }
            }
          }
        }

        onBeginContact(selfCollider, otherCollider, contact) {
          // 只在两个碰撞体开始接触时被调用一次
          console.log('onBeginContact');

          if (selfCollider.node.name == "box" || otherCollider.node.name == "box") {
            (_crd && mainScene === void 0 ? (_reportPossibleCrUseOfmainScene({
              error: Error()
            }), mainScene) : mainScene).Instance.winNode.active = true;
            return;
          }

          this.war = true;
          selfCollider.node.getComponent(soldier).speed = 0;
          otherCollider.node.getComponent(soldier).speed = 0;

          if (selfCollider.node.name == otherCollider.node.name) {
            return;
          }

          this.selfCollider = selfCollider;
          this.otherCollider = otherCollider;
          (_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
            error: Error()
          }), Gloabl) : Gloabl).speed = 0;
          var self_animationComponent = selfCollider.node.getComponent(Animation);
          self_animationComponent.play(self_animationComponent.clips[2].name);
          var other_animationComponent = otherCollider.node.getComponent(Animation);
          other_animationComponent.play(other_animationComponent.clips[2].name);

          if ((_crd && mainScene === void 0 ? (_reportPossibleCrUseOfmainScene({
            error: Error()
          }), mainScene) : mainScene).js == true) {
            self_animationComponent.getState(self_animationComponent.clips[2].name).speed = 1.5;
            other_animationComponent.getState(other_animationComponent.clips[2].name).speed = 1.5;
          }

          this.num = 0;
        }

        aniFightEnd() {
          this.selfCollider.node.getComponent(soldier).hp -= this.otherCollider.node.getComponent(soldier).fight;
          this.otherCollider.node.getComponent(soldier).hp -= this.selfCollider.node.getComponent(soldier).fight;
          console.log(this.selfCollider.node.getComponent(soldier).hp + " " + this.otherCollider.node.getComponent(soldier).hp);
          var self_animationComponent = this.selfCollider.node.getComponent(Animation);
          var other_animationComponent = this.otherCollider.node.getComponent(Animation);

          if (this.selfCollider.node.getComponent(soldier).hp <= 0 && this.otherCollider.node.getComponent(soldier).hp <= 0) {
            this.selfCollider.enabled = false;
            this.otherCollider.enabled = false;
            self_animationComponent.play(self_animationComponent.clips[1].name);
            other_animationComponent.play(other_animationComponent.clips[1].name);

            if ((_crd && mainScene === void 0 ? (_reportPossibleCrUseOfmainScene({
              error: Error()
            }), mainScene) : mainScene).js == true) {
              self_animationComponent.getState(self_animationComponent.clips[1].name).speed = 1.5;
              other_animationComponent.getState(other_animationComponent.clips[1].name).speed = 1.5;
            }
          } else if (this.selfCollider.node.getComponent(soldier).hp <= 0) {
            this.selfCollider.enabled = false; //this.selfCollider.destroy();

            self_animationComponent.play(self_animationComponent.clips[1].name);
            other_animationComponent.play(other_animationComponent.clips[0].name);
            this.war = false;

            if ((_crd && mainScene === void 0 ? (_reportPossibleCrUseOfmainScene({
              error: Error()
            }), mainScene) : mainScene).js == true) {
              self_animationComponent.getState(self_animationComponent.clips[1].name).speed = 1.5;
              other_animationComponent.getState(other_animationComponent.clips[0].name).speed = 1.5;
            }
          } else if (this.otherCollider.node.getComponent(soldier).hp <= 0) {
            this.otherCollider.enabled = false;
            other_animationComponent.play(other_animationComponent.clips[1].name);
            self_animationComponent.play(self_animationComponent.clips[0].name);
            this.war = false; //this.otherCollider.destroy();

            if ((_crd && mainScene === void 0 ? (_reportPossibleCrUseOfmainScene({
              error: Error()
            }), mainScene) : mainScene).js == true) {
              self_animationComponent.getState(self_animationComponent.clips[1].name).speed = 1.5;
              other_animationComponent.getState(other_animationComponent.clips[0].name).speed = 1.5;
            }
          } else {
            self_animationComponent.play(self_animationComponent.clips[2].name);
            other_animationComponent.play(other_animationComponent.clips[2].name);

            if ((_crd && mainScene === void 0 ? (_reportPossibleCrUseOfmainScene({
              error: Error()
            }), mainScene) : mainScene).js == true) {
              self_animationComponent.getState(self_animationComponent.clips[2].name).speed = 1.5;
              other_animationComponent.getState(other_animationComponent.clips[2].name).speed = 1.5;
            }
          }

          ;
        }

        aniDeathEnd() {
          var self_animationComponent = this.selfCollider.node.getComponent(Animation);
          var other_animationComponent = this.otherCollider.node.getComponent(Animation);

          if (this.selfCollider.node.getComponent(soldier).hp <= 0) {
            if (this.selfCollider.node.name == "enemy") {
              if (this.selfCollider.node.getComponent(BoxCollider2D).tag == 1) {
                (_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
                  error: Error()
                }), Gloabl) : Gloabl).coin += 5;
              } else if (this.selfCollider.node.getComponent(BoxCollider2D).tag == 2) {
                (_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
                  error: Error()
                }), Gloabl) : Gloabl).coin += 10;
              } else if (this.selfCollider.node.getComponent(BoxCollider2D).tag == 3) {
                (_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
                  error: Error()
                }), Gloabl) : Gloabl).coin += 30;
              }
            }

            this.selfCollider.node.active = false;
            this.selfCollider.node.removeFromParent();
            this.speed = 100;
            (_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
              error: Error()
            }), Gloabl) : Gloabl).speed = 100;
            (_crd && mainScene === void 0 ? (_reportPossibleCrUseOfmainScene({
              error: Error()
            }), mainScene) : mainScene).Instance.refresh();
          }

          if (this.otherCollider.node.getComponent(soldier).hp <= 0) {
            this.speed = 100;
            (_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
              error: Error()
            }), Gloabl) : Gloabl).speed = 100;

            if (this.otherCollider.node.name == "enemy") {
              if (this.otherCollider.node.getComponent(BoxCollider2D).tag == 1) {
                (_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
                  error: Error()
                }), Gloabl) : Gloabl).coin += 5;
              } else if (this.otherCollider.node.getComponent(BoxCollider2D).tag == 2) {
                (_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
                  error: Error()
                }), Gloabl) : Gloabl).coin += 10;
              } else if (this.otherCollider.node.getComponent(BoxCollider2D).tag == 3) {
                (_crd && Gloabl === void 0 ? (_reportPossibleCrUseOfGloabl({
                  error: Error()
                }), Gloabl) : Gloabl).coin += 30;
              }
            }

            this.otherCollider.node.active = false;
            this.otherCollider.node.removeFromParent();
            (_crd && mainScene === void 0 ? (_reportPossibleCrUseOfmainScene({
              error: Error()
            }), mainScene) : mainScene).Instance.refresh();
          } // else{
          //     self_animationComponent.play(self_animationComponent.clips[2].name);
          //     other_animationComponent.play(other_animationComponent.clips[2].name);
          // }

        }

      }, _temp), (_descriptor = _applyDecoratedDescriptor(_class2.prototype, "hp", [property], {
        configurable: true,
        enumerable: true,
        writable: true,
        initializer: function initializer() {
          return 0;
        }
      }), _descriptor2 = _applyDecoratedDescriptor(_class2.prototype, "fight", [property], {
        configurable: true,
        enumerable: true,
        writable: true,
        initializer: function initializer() {
          return 0;
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
//# sourceMappingURL=soldier.js.map